import unittest
import torch
from torch import nn
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

class TestVariationalInference(unittest.TestCase):

    def test_kl_divergence(self):
        #create model
        #do two inferences over same datapoint, check if different

        to_feed = torch.ones((1, 10))

        @variational_estimator
        class VariationalEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.blinear = BayesianLinear(10, 10)

            def forward(self, x):
                return self.blinear(x)

        model = VariationalEstimator()
        predicted = model(to_feed)

        complexity_cost = model.nn_kl_divergence()
        kl_complexity_cost = kl_divergence_from_nn(model)

        self.assertEqual((complexity_cost == kl_complexity_cost).all(), torch.tensor(True))

    def test_variance_over_prediction(self):
        #create model
        #do two inferences over model
        #check if able to gather variance over predictions
        pass

    def test_freeze_estimator(self):
        #create model, freeze it
        #infer two times on same datapoint, check if all equal
        pass

if __name__ == "__main__":
    unittest.main()