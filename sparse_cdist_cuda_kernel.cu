#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


#include "math.h"


template <typename scalar_t> __device__ void cpy_array(scalar_t* from, scalar_t* to, int start, int end)
{
  int counter = 0;
  for (int i=start; i<end; i++){
    to[counter]=from[i];
    counter++;
  }
}




template <typename scalar_t>
__global__ void sparse_cdist_cuda_kernel(
    const int64_t* __restrict__ a_rowptr,
    const int64_t* __restrict__ a_col,
    scalar_t* __restrict__ a_value,
    int64_t* __restrict__ b_rowptr,
    int64_t* __restrict__ b_col,
    scalar_t* __restrict__ b_value,
    scalar_t* __restrict__ output,
    int dim_a,
    int dim_b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < dim_a && j < dim_b){
    const int start_i = a_rowptr[i];
    const int end_i = a_rowptr[i+1];
    const int start_j = b_rowptr[j];
    const int end_j = b_rowptr[j+1];

    scalar_t distance = 0.0;

    scalar_t *b_value_remainder = new scalar_t[end_j-start_j];
    cpy_array<scalar_t>(b_value, b_value_remainder, start_j, end_j);

    for (int ii = start_i; ii < end_i; ii ++){
      int col_index_i = a_col[ii];
      auto value_i = a_value[ii];
      bool not_matched_i = true;
      int counter = 0;
      for (int jj = start_j; jj < end_j; jj ++){
        int col_index_j = b_col[jj];
        auto value_j = b_value[jj];

        if (col_index_i == col_index_j){
          auto t = (value_i - value_j);
          t *=t;
          distance += t;
          not_matched_i = false;
          b_value_remainder[counter] = 0.0;
        }
        counter++;
      }
      if(not_matched_i){
        distance +=(value_i*value_i);
      }
    }
    for (int jj = 0; jj < end_j- start_j; jj ++){
      distance +=(b_value_remainder[jj]*b_value_remainder[jj]);
    }
    distance = sqrt(distance);
    output[i*dim_b + j] = distance;

  }
}


template <typename scalar_t>
__global__ void sparse_cdist_bw_cuda_kernel(
    const int64_t* __restrict__ a_row,
    const int64_t* __restrict__ a_col,
    scalar_t* __restrict__ a_value,
    int64_t* __restrict__ b_row,
    int64_t* __restrict__ b_col,
    scalar_t* __restrict__ b_value,
    scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ distances,
    scalar_t* __restrict__ grad,
    int b_col_size,
    int dim_distance_a
  ) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = blockIdx.y * blockDim.y + threadIdx.y;

  auto a_mn_value_grad = a_value[m];
  
  for(int j=0; j < b_col_size, j++){
    if(b_col == n){
      scalar_t b_val_jn= b_value[j];
      scalar_t aggregator = 0;
      for(int i=0; i < dim_distance_a, i++){
        aggregator += grad_out[i*dim_distance_a + j] * distances[i*dim_distance_a + j];
      }
      a_mn_value_grad -= b_val_jn * aggregator;
    }
  }
  grad[m] = a_mn_value_grad;
}




at::Tensor sparse_cdist_cuda(
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    int dim_a,
    int dim_b
    ) {

  std::vector<int64_t> vec;
  vec.push_back(dim_a);
  vec.push_back(dim_b);
  auto options = a_value_data.options();
  torch::Tensor output = torch::zeros(vec,options = options);

  
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(a_rowptr_data.size(0)+1 / threadsPerBlock.x, b_rowptr_data.size(0)+1 / threadsPerBlock.y);
  AT_DISPATCH_FLOATING_TYPES(a_value_data.scalar_type(), "sparse_cdist_cuda", ([&] {
    sparse_cdist_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        a_rowptr_data.data_ptr<int64_t>(),
        a_col_data.data_ptr<int64_t>(),
        a_value_data.data_ptr<scalar_t>(),
        b_rowptr_data.data_ptr<int64_t>(),
        b_col_data.data_ptr<int64_t>(),
        b_value_data.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        a_rowptr_data.size(0),
        b_rowptr_data.size(0));

  }));

  return output;
}

torch::Tensor sparse_cdist_bw_cuda(
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    torch::Tensor grad_out,
    torch::Tensor distance,
    int dim_a,
    int dim_b
    ) {

  torch::Tensor grad_a = torch::zeros_like(grad_out, grad_out.options());
  std::cout << "grad_a after creation is: " << grad_a;

  
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(a_rowptr_data.size(0)+1 / threadsPerBlock.x, b_rowptr_data.size(0)+1 / threadsPerBlock.y);
  AT_DISPATCH_FLOATING_TYPES(a_value_data.scalar_type(), "sparse_cdist_bw_cuda", ([&] {
    sparse_cdist_bw_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        a_rowptr_data.data_ptr<int64_t>(),
        a_col_data.data_ptr<int64_t>(),
        a_value_data.data_ptr<scalar_t>(),
        b_rowptr_data.data_ptr<int64_t>(),
        b_col_data.data_ptr<int64_t>(),
        b_value_data.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        distance.data_ptr<scalar_t>(),
        grad_a.data_ptr<scalar_t>(),
        a_rowptr_data.size(0),
        b_rowptr_data.size(0));

  }));

  return grad_a;
}