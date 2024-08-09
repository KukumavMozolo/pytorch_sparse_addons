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
    const int64_t* __restrict__ a_rowptr,
    const int64_t* __restrict__ a_col,
    scalar_t* __restrict__ a_value,
    int64_t* __restrict__ b_rowptr,
    int64_t* __restrict__ b_col,
    scalar_t* __restrict__ b_value,
    scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ distances,
    scalar_t* __restrict__ grad,
    int dim_a,
    int dim_b,
    int dim_c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < dim_a && j < dim_b){
    const int start_i = a_rowptr[i];
    const int end_i = a_rowptr[i+1];
    const int start_j = b_rowptr[j];
    const int end_j = b_rowptr[j+1];


    scalar_t *b_value_remainder = new scalar_t[end_j-start_j];
    cpy_array<scalar_t>(b_value, b_value_remainder, start_j, end_j);
    int64_t *b_col_remainder = new int64_t[end_j-start_j];
    cpy_array<int64_t>(b_col, b_col_remainder, start_j, end_j);

    for (int ii = start_i; ii < end_i; ii ++){
      int col_index_i = a_col[ii];
      auto value_i = a_value[ii];
      bool not_matched_i = true;
      int counter = 0;
      for (int jj = start_j; jj < end_j; jj ++){
        int col_index_j = b_col[jj];
        auto value_j = b_value[jj];

        if (col_index_i == col_index_j){
          auto delta = (value_i - value_j);
          if (distances[i*dim_b + j] != 0){
            grad[col_index_i * dim_a * dim_b + i * dim_b + j] = (delta/distances[i*dim_b + j]) * grad_out[i * dim_b + j];
          }
          
          not_matched_i = false;
          b_value_remainder[counter] = 0.0;
        }
        counter++;
      }
      if(not_matched_i){
        grad[col_index_i * dim_a * dim_b + i * dim_b + j] = (value_i/distances[i*dim_b + j]) * grad_out[i * dim_b + j];
      }
    }
    for (int jj = 0; jj < end_j- start_j; jj ++){
      if (b_value_remainder !=0){
      int64_t col_index = b_col_remainder[jj];
      grad[col_index * dim_a * dim_b + i * dim_b + j] = ((-1.0*b_value_remainder[jj])/distances[i*dim_b + j]) * grad_out[i * dim_b + j];
      }
    }
  }
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
  dim3 numBlocks(dim_a+1 / threadsPerBlock.x, dim_b+1 / threadsPerBlock.y);
  AT_DISPATCH_FLOATING_TYPES(a_value_data.scalar_type(), "sparse_cdist_cuda", ([&] {
    sparse_cdist_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        a_rowptr_data.data_ptr<int64_t>(),
        a_col_data.data_ptr<int64_t>(),
        a_value_data.data_ptr<scalar_t>(),
        b_rowptr_data.data_ptr<int64_t>(),
        b_col_data.data_ptr<int64_t>(),
        b_value_data.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        dim_a,
        dim_b);

  }));

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> sparse_cdist_bw_cuda(
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    torch::Tensor grad_out,
    torch::Tensor distance,
    int dim_a,
    int dim_b,
    int dim_c
    ) {

  std::vector<int64_t> vec_a;
  vec_a.push_back(dim_a);
  vec_a.push_back(dim_b);
  vec_a.push_back(dim_c);
  torch::Tensor grad_a = torch::zeros(vec_a, a_value_data.options());
  
  dim3 threadsPerBlockA(32, 32);
  dim3 numBlocksA(dim_a+1 / threadsPerBlockA.x, dim_b+1 / threadsPerBlockA.y);
  AT_DISPATCH_FLOATING_TYPES(a_value_data.scalar_type(), "sparse_cdist_bw_cuda", ([&] {
    sparse_cdist_bw_cuda_kernel<scalar_t><<<numBlocksA, threadsPerBlockA>>>(
        a_rowptr_data.data_ptr<int64_t>(),
        a_col_data.data_ptr<int64_t>(),
        a_value_data.data_ptr<scalar_t>(),
        b_rowptr_data.data_ptr<int64_t>(),
        b_col_data.data_ptr<int64_t>(),
        b_value_data.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        distance.data_ptr<scalar_t>(),
        grad_a.data_ptr<scalar_t>(),
        dim_a,
        dim_b,
        dim_c);

  }));

  std::vector<int64_t> vec_b;
  vec_b.push_back(dim_b);
  vec_b.push_back(dim_a);
  vec_b.push_back(dim_c);
  torch::Tensor grad_b = torch::zeros(vec_b, a_value_data.options());

  dim3 threadsPerBlockB(32, 32);
  dim3 numBlocksB(dim_a+1 / threadsPerBlockB.x, dim_b+1 / threadsPerBlockB.y);
  AT_DISPATCH_FLOATING_TYPES(a_value_data.scalar_type(), "sparse_cdist_bw_cuda", ([&] {
    sparse_cdist_bw_cuda_kernel<scalar_t><<<numBlocksB, threadsPerBlockB>>>(
        a_rowptr_data.data_ptr<int64_t>(),
        a_col_data.data_ptr<int64_t>(),
        a_value_data.data_ptr<scalar_t>(),
        b_rowptr_data.data_ptr<int64_t>(),
        b_col_data.data_ptr<int64_t>(),
        b_value_data.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        distance.data_ptr<scalar_t>(),
        grad_b.data_ptr<scalar_t>(),
        dim_b,
        dim_a,
        dim_c);

  }));  

  return std::make_tuple(grad_a, grad_b);
}