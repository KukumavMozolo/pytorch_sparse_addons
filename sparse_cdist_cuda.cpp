#include <torch/extension.h>


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define IS_CUDA(x) x.device().is_cuda()

torch::Tensor sparse_cdist_cuda(
  torch::Tensor a_rowptr_data,
  torch::Tensor a_col_data,
  torch::Tensor a_value_data,
  torch::Tensor b_rowptr_data,
  torch::Tensor b_col_data,
  torch::Tensor b_value_data,
  int dim_a,
  int dim_b);



template <typename scalar_t> void cpy_array_cpu(const scalar_t* from, scalar_t* to, int start, int end)
{
  int counter = 0;
  for (int i=start; i<end; i++){
    to[counter]=from[i];
    counter++;
  }
}


template <typename scalar_t>
void sparse_cdist_cpu(
    const int64_t* __restrict__ a_rowptr,
    const int64_t* __restrict__ a_col,
    const scalar_t* __restrict__ a_value,
    const int64_t* __restrict__ b_rowptr,
    const int64_t* __restrict__ b_col,
    const scalar_t* __restrict__ b_value,
    scalar_t* __restrict__ output,
  int dim_a,
  int dim_b){
    for (int i=0; i<dim_a; i++){
      for (int j=0; j<dim_b; j++){
        const int start_i = a_rowptr[i];
        const int end_i = a_rowptr[i+1];
        const int start_j = b_rowptr[j];
        const int end_j = b_rowptr[j+1];

        scalar_t distance = 0.0;

        scalar_t *b_value_remainder = new scalar_t[end_j-start_j];
        cpy_array_cpu<scalar_t>(b_value, b_value_remainder, start_j, end_j);

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
      delete[] b_value_remainder;
      }
    } 
  }

torch::Tensor sparse_cdist_cpu_switch(
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    int dim_a,
    int dim_b)
  {
    std::vector<int64_t> vec;
    vec.push_back(dim_a);
    vec.push_back(dim_b);
    auto options = a_value_data.options();
    torch::Tensor output = torch::zeros(vec,options = options);
    switch (a_value_data.scalar_type()) {
      case at::ScalarType::Double:
        sparse_cdist_cpu<double>(a_rowptr_data.data_ptr<int64_t>(), a_col_data.data_ptr<int64_t>(), a_value_data.data_ptr<double>(), b_rowptr_data.data_ptr<int64_t>(), b_col_data.data_ptr<int64_t>(), b_value_data.data_ptr<double>(),output.data_ptr<double>(), dim_a, dim_b);
        break;
      case at::ScalarType::Float:
        sparse_cdist_cpu<float>(a_rowptr_data.data_ptr<int64_t>(), a_col_data.data_ptr<int64_t>(), a_value_data.data_ptr<float>(), b_rowptr_data.data_ptr<int64_t>(), b_col_data.data_ptr<int64_t>(), b_value_data.data_ptr<float>(),output.data_ptr<float>(), dim_a, dim_b);            
        break;
      default: TORCH_CHECK(false, "Only sparse float tensors are supported!");
    }
    return output;
  }




torch::Tensor sparse_cdist(
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    int dim_a,
    int dim_b){
      CHECK_CONTIGUOUS(a_rowptr_data);
      CHECK_CONTIGUOUS(a_col_data);
      CHECK_CONTIGUOUS(a_value_data);
      CHECK_CONTIGUOUS(b_rowptr_data);
      CHECK_CONTIGUOUS(b_col_data);
      CHECK_CONTIGUOUS(b_value_data);
      if(!IS_CUDA(a_rowptr_data) && !IS_CUDA(a_rowptr_data) && !IS_CUDA(a_rowptr_data) && !IS_CUDA(b_rowptr_data) && !IS_CUDA(b_col_data) && !IS_CUDA(b_value_data)){
        return sparse_cdist_cpu_switch(a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data, dim_a, dim_b);
      }else if (IS_CUDA(a_rowptr_data) && IS_CUDA(a_rowptr_data) && IS_CUDA(a_rowptr_data) && IS_CUDA(b_rowptr_data) && IS_CUDA(b_col_data) && IS_CUDA(b_value_data)){
        return sparse_cdist_cuda(a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data, dim_a, dim_b);
      }
      else{
        TORCH_CHECK(false, "All Tensors must be on same device!");
      }
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cdist", &sparse_cdist, "Sparse Cdist (CUDA)");
}

