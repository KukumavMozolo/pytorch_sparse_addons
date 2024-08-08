import unittest
import torch
import numpy as np

from torch_sparse import SparseTensor

from torch_sparse_addons.dist import cdist


class CdistTest(unittest.TestCase):
    def test_cdist(self):
        x = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0]]))
        res = cdist(x)
        target = torch.cdist(x.to_dense(),x.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().numpy(), target.to_dense().numpy()))

        y = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0]]))
        res2 = cdist(x,y)
        target2 = torch.cdist(x.to_dense(),y.to_dense())
        self.assertTrue(np.array_equal(res2.to_dense().numpy(), target2.to_dense().numpy()))


if __name__ == "__main__":
    unittest.main()