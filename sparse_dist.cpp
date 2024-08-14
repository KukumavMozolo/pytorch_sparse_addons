
#include <torch/extension.h>

#define SPARSE_API

torch::Tensor sparse_cdist(
  torch::Tensor a_rowptr_data,
  torch::Tensor a_col_data,
  torch::Tensor a_value_data,
  torch::Tensor b_rowptr_data,
  torch::Tensor b_col_data,
  torch::Tensor b_value_data,
  int dim_a,
  int dim_b);

torch::Tensor sparse_bw_cdist(
  torch::Tensor a_rowptr_data,
  torch::Tensor a_col_data,
  torch::Tensor a_value_data,
  torch::Tensor b_rowptr_data,
  torch::Tensor b_col_data,
  torch::Tensor b_value_data,
  torch::Tensor grad_out,
  torch::Tensor distance,
  int dim_a,
  int dim_b);  

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SparseCdist : public torch::autograd::Function<SparseCdist> {
public: static variable_list forward(
    AutogradContext *ctx,
    torch::Tensor a_rowptr_data,
    torch::Tensor a_col_data,
    torch::Tensor a_value_data,
    torch::Tensor b_rowptr_data,
    torch::Tensor b_col_data,
    torch::Tensor b_value_data,
    int dim_a,
    int dim_b
    ) {
    auto out = sparse_cdist(a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data, dim_a, dim_b);
    ctx->saved_data["dim_a"] = dim_a;
    ctx->saved_data["dim_b"] = dim_b;
    ctx->save_for_backward({a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data, out});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto dim_a = ctx->saved_data["dim_a"].toInt();
    auto dim_b = ctx->saved_data["dim_b"].toInt();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto a_rowptr_data = saved[0], a_col_data = saved[1], a_value_data = saved[2], b_rowptr_data = saved[3],
         b_col_data = saved[4], b_value_data = saved[5], distance = saved[5];

    auto grad_value_a = Variable();
    if (torch::autograd::any_variable_requires_grad({a_value_data})){
      std::cout << "grad_outs is: " << grad_out;
      grad_value_a = sparse_bw_cdist(a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data, grad_out, distance, dim_a, dim_b);
    }
    
    auto grad_value_b = Variable();
    if (torch::autograd::any_variable_requires_grad({b_value_data})){
      std::cout << "grad_outs is: " << grad_out;
      grad_value_b = sparse_bw_cdist(b_rowptr_data, b_col_data, b_value_data,a_rowptr_data, a_col_data, a_value_data, grad_out, distance, dim_b, dim_a);
    }    
    return {Variable(), Variable(), grad_value_a,
            Variable(), Variable(), grad_value_b, Variable(), Variable(), Variable()};
  }
};

SPARSE_API torch::Tensor cdist(    
  torch::Tensor a_rowptr_data,
  torch::Tensor a_col_data,
  torch::Tensor a_value_data,
  torch::Tensor b_rowptr_data,
  torch::Tensor b_col_data,
  torch::Tensor b_value_data,
  int dim_a,
  int dim_b) {
  return SparseCdist::apply(a_rowptr_data, a_col_data, a_value_data, b_rowptr_data, b_col_data, b_value_data,
                        dim_a, dim_b)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cdist", &cdist, "Sparse Cdist (CUDA)");
}