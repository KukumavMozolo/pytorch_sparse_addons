import torch
from sparse_cdist import cdist as cuda_cdist
from torch_sparse import SparseTensor


def cdist(x: SparseTensor, y: SparseTensor = None):
    x_rowptr, x_col, x_value =  x.csr()
    if y is None:
        return cuda_cdist(x_rowptr, x_col,x_value, x_rowptr,x_col,x_value, x.size(0), x.size(0))
    else:
        assert x.size(1) == y.size(1)
        y_rowptr, y_col, y_value =  y.csr()
        return cuda_cdist(x_rowptr, x_col,x_value, y_rowptr,y_col,y_value, x.size(0), y.size(0))