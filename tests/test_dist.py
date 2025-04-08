import unittest
import torch
import numpy as np
from torch_sparse import SparseTensor

from pytorch_sparse_addons.dist import cdist
from time import sleep



class CdistTest(unittest.TestCase):
    def test_cdist(self):
        simple = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0]], device=torch.device("cuda")))
        res = cdist(simple)
        target = torch.cdist(simple.to_dense(),simple.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))

        simple2 = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0], [2.0,4.0,6.0]], device=torch.device("cuda")))
        res = cdist(simple, simple2)
        target = torch.cdist(simple.to_dense(),simple2.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))        

        simple3 = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0], [2.0,4.0,0.0]], device=torch.device("cuda")))
        res = cdist(simple, simple3)
        target = torch.cdist(simple.to_dense(),simple3.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))

        simple4 = SparseTensor.from_dense(torch.tensor([[2.0,0.0,0.0]], device=torch.device("cuda")))
        res = cdist(simple4, simple3)
        target = torch.cdist(simple4.to_dense(),simple3.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))     

        res = cdist(simple3, simple4)
        target = torch.cdist(simple3.to_dense(),simple4.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))             

        x = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,1.0]], device=torch.device("cuda")))
        res = cdist(x)
        target = torch.cdist(x.to_dense(),x.to_dense())
        self.assertTrue(np.array_equal(res.to_dense().cpu().numpy(), target.to_dense().cpu().numpy()))

        y = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0],[.0,1.0,2.0]], device=torch.device("cuda")))
        res2 = cdist(x,y)
        target2 = torch.cdist(x.to_dense(),y.to_dense())
        self.assertTrue(np.array_equal(res2.to_dense().cpu().numpy(), target2.to_dense().cpu().numpy()))

        z = SparseTensor.from_dense(torch.tensor([[1.0,2.0,],[1.0,.0]], device=torch.device("cuda")))
        self.assertRaises(AssertionError,lambda: cdist(x,z))

        j = SparseTensor.from_dense(torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0],[.0,1.0,2.0]]))
        self.assertRaises(RuntimeError,lambda: cdist(x,j))

        k = torch.tensor([[1.0,2.0,3.0],[1.0,.0,.0],[.0,1.0,.0],[.0,1.0,2.0],[.0,1.0,2.0]])
        self.assertRaises(AttributeError,lambda: cdist(x,k))

        l = SparseTensor.from_dense(torch.tensor([[1,2,3],[1,0,0],[0,1,0],[0,1,2],[0,1,2]], device=torch.device("cuda")))
        self.assertRaises(RuntimeError,lambda: cdist(x,l))

if __name__ == "__main__":
    unittest.main()