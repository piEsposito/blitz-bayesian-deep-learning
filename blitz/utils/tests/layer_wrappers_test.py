import unittest
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianConv2d, BayesianLinear, BayesianLSTM, BayesianEmbedding, GaussianVariational, BayesianGRU
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from blitz.utils import Flipout, Radial

class TestFlipout(unittest.TestCase):

    def test_linear(self):
        layer = Flipout(BayesianLinear)(10, 10)
        in_ = torch.ones(2, 10)
        out_ = layer(in_)
        #print(out_)
        self.assertEqual((out_[0,:] != out_[1,:]).any(), torch.tensor(True))

    def test_RNN(self):
        layer = Flipout(BayesianLSTM)(10, 10)
        in_ = torch.ones(2, 3, 10)
        out_, _ = layer(in_)
        #print(out_)
        self.assertEqual((out_[0,:,:] != out_[1,:,:]).any(), torch.tensor(True))

class TestRadial(unittest.TestCase):

    def test_linear(self):
        layer = Radial(BayesianLinear)(10, 10)
        in_ = torch.ones(2, 10)
        out_ = layer(in_)
        #print(out_)

    def test_RNN(self):
        layer = Radial(BayesianLSTM)(10, 10)
        layer2 = Radial(BayesianGRU)(10, 10)
        in_ = torch.ones(2, 3, 10)
        out_, _ = layer(in_)
        our_, _ = layer2(in_)
        #print(out_)

class TestNested(unittest.TestCase):

    def test_linear(self):
        layer = Radial(Flipout(BayesianLinear)(10, 10))
        in_ = torch.ones(2, 10)
        out_ = layer(in_)
        #print(out_)
        self.assertEqual((out_[0,:] != out_[1,:]).any(), torch.tensor(True))

    def test_RNN(self):
        layer = Radial(Flipout(BayesianLSTM)(10, 10))
        in_ = torch.ones(2, 3, 10)
        out_, _ = layer(in_)
        #print(out_)
        self.assertEqual((out_[0,:,:] != out_[1,:,:]).any(), torch.tensor(True))

if __name__ == "__main__":
    unittest.main()