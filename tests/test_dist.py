import unittest
import torch
import numpy as np
from torch_sparse import SparseTensor
import sys
import os
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from src.pytorch_sparse_addons.dist import cdist


class CdistTest(unittest.TestCase):
    def test_cdist(self):
        x = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,1.0]]))
        res = cdist(x)
        target = torch.cdist(x.to_dense(),x.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().numpy(), target.to_dense().numpy()))

        y = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0]]))
        res2 = cdist(x,y)
        target2 = torch.cdist(x.to_dense(),y.to_dense())
        self.assertTrue(np.array_equal(res2.to_dense().numpy(), target2.to_dense().numpy()))

    def test_cdist_grad(self):
        x = torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0]],requires_grad=True)
        x22 = torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0]],requires_grad=True)
        dist_x = torch.cdist(x,x)
        dist_y = torch.tensor([[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]])
        loss = torch.nn.MSELoss()
        output = loss(dist_x, dist_y)
        output.backward()
        print("")

        x2 = SparseTensor.from_dense(x22)
        dist_x2 = cdist(x2,x2)
        dist_y2 = torch.tensor([[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]])
        loss = torch.nn.MSELoss()
        output = loss(dist_x2, dist_y2)
        output.backward()
        print("")


if __name__ == "__main__":
    unittest.main()