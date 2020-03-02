import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bbp_torch.losses import kl_divergence_from_nn
from bbp_torch.modules import BayesianLinear

class TestKLDivergence(unittest.TestCase):

    def test_kl_divergence_bayesian_module(self):
        blinear = BayesianLinear(10, 10)
        to_feed = torch.ones((1, 10))
        predicted = blinear(to_feed)

        complexity_cost = blinear.log_variational_posterior - blinear.log_prior
        kl_complexity_cost = kl_divergence_from_nn(blinear)

        self.assertEqual((complexity_cost == kl_complexity_cost).all(), torch.tensor(True))
        pass
    
    def test_kl_divergence_non_bayesian_module(self):
        linear = nn.Linear(10, 10)
        to_feed = torch.ones((1, 10))
        predicted = linear(to_feed)

        kl_complexity_cost = kl_divergence_from_nn(linear)
        self.assertEqual((torch.tensor(0) == kl_complexity_cost).all(), torch.tensor(True))
        pass

    def test_kl_divergence_whole_model(self):
        pass

if __name__ == "__main__":
    unittest.main()