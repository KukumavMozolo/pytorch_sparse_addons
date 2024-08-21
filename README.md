# pytorch_sparse_addons

# Does not support autograd atm
## Requires
- python==3.11
- torch==2.3.0
- torch_scatter
- torch_sparse

## Usage
```python
from pytorch_sparse_addons.dist import cdist
x = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,1.0]]))
y = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0],[.0,1.0,2.0]]))
result = cdist(x,y)
```

## Building
Build for compute capability 8.0, 8.6, 8.9
to support your cards compute capability e.g. build with `TORCH_CUDA_ARCH_LIST="<compute_capability;other_compute_capability>" pdm build`