#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "math.h"


template <typename scalar_t> __device__ void cpy_array(const scalar_t* from, scalar_t* to, int start, int end)
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
    const scalar_t* __restrict__ a_value,
    const int64_t* __restrict__ b_rowptr,
    const int64_t* __restrict__ b_col,
    const scalar_t* __restrict__ b_value,
    scalar_t* __restrict__ output,
    int dim_a,
    int dim_b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < dim_a && j < dim_b){
    // printf("dima_a: %d,i: %d,dim_b: %d,j: %d \n", dim_a, i, dim_b, j);
    const int start_i = a_rowptr[i];
    const int end_i = a_rowptr[i+1];
    int start_j = b_rowptr[j];
    const int end_j = b_rowptr[j+1];

    scalar_t distance = 0.0;

    if(start_j == end_j){
      for (int ii = start_i; ii < end_i; ii ++){
        auto value_i = a_value[ii];
        distance +=(value_i*value_i);
      }
    }
    else if(start_i == end_i){
      for (int jj = start_j; jj < end_j; jj ++){
        auto value_j = b_value[jj];
        distance +=(value_j*value_j);
      }
    }
    else{
      int col_index_j = b_col[start_j];
      auto value_j = b_value[start_j];
      bool j_empty = false;
      //printf("loop from start_i: %d, to end_i: %d \n", start_i, end_i);
      for (int ii = start_i; ii < end_i; ii ++){
        int col_index_i = a_col[ii];
        auto value_i = a_value[ii];

        if (col_index_i == col_index_j && !j_empty){
            auto t = (value_i - value_j);
            //printf("t: %f value_i: %f, value_j: %f\n", t, value_i, value_j);
            distance += t*t;
            start_j++;
            if(start_j < end_j){
              col_index_j = b_col[start_j];
              value_j = b_value[start_j];  
            }
            else{
              j_empty = true;
            }
            //printf("same: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, col_index_i, col_index_j);
        }
        else if(col_index_i< col_index_j){
          distance +=(value_i*value_i);
          //printf("smaller: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, col_index_i, col_index_j);
        }
        else{
          while (col_index_j < col_index_i && !j_empty){
              distance +=(value_j*value_j);
              start_j++;
              if(start_j < end_j){
                col_index_j = b_col[start_j];
                value_j = b_value[start_j];  
              }else{
                j_empty = true;
              }
            //printf("else larger: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, col_index_i, col_index_j);          
          }
          if (col_index_i == col_index_j && !j_empty){
              auto t = (value_i - value_j);
              //printf("t: %f value_i: %f, value_j: %f\n", t, value_i, value_j);
              distance += t*t;
              start_j++;
              if(start_j < end_j){
                col_index_j = b_col[start_j];
                value_j = b_value[start_j];  
              }
              else{
                j_empty = true;
              }
              //printf("else same: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, col_index_i, col_index_j);
          }
          else{
            distance +=(value_i*value_i);
            //printf("else smaller: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, col_index_i, col_index_j);
          }
        }
      }
      if(!j_empty){
        for(int jj=start_j; jj<end_j; jj++){
          value_j = b_value[jj];
          col_index_j = b_col[jj];
          distance +=(value_j*value_j);
          //printf("rest: distance: %f,col_index_i: %d,col_index_j: %d \n", distance, 0, col_index_j);
        }
      }

      distance = sqrt(distance);
      //printf("rest: distance: %f\n", distance);
      output[i*dim_b + j] = distance;
    }
  }
}


torch::Tensor sparse_cdist_cuda(
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