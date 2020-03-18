import unittest
import torch
from torch import nn

from blitz.modules import BayesianLinear
from blitz.modules.base_bayesian_module import BayesianModule

class TestLinearBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        module = BayesianLinear(10, 10)
        pass

    def test_infer_shape_1_sample(self):
        blinear = BayesianLinear(10, 10)
        linear = nn.Linear(10, 10)
        to_feed = torch.ones((1, 10))

        b_infer = linear(to_feed)
        l_infer = blinear(to_feed)

        self.assertEqual(b_infer.shape, l_infer.shape)
        self.assertEqual((b_infer == b_infer).all(), torch.tensor(True))
        pass

    def test_variational_inference(self):
        #create module, check if inference is variating
        blinear = BayesianLinear(10, 10)
        linear = nn.Linear(10, 10)

        to_feed = torch.ones((1, 10))
        self.assertEqual((blinear(to_feed) != blinear(to_feed)).any(), torch.tensor(True))
        self.assertEqual((linear(to_feed) == linear(to_feed)).all(), torch.tensor(True))
        pass

    def test_freeze_module(self):
        #create module, freeze
        #check if two inferences keep equal
        blinear = BayesianLinear(10, 10)
        to_feed = torch.ones((1, 10))
        self.assertEqual((blinear(to_feed) != blinear(to_feed)).any(), torch.tensor(True))

        frozen_feedforward = blinear.forward_frozen(to_feed)
        blinear.freeze = True
        self.assertEqual((blinear.forward(to_feed) == frozen_feedforward).all(), torch.tensor(True))

    def test_no_bias(self):
        blinear = BayesianLinear(10, 10, bias=False)
        to_feed = torch.ones((1, 10))
        self.assertEqual((blinear(to_feed) != blinear(to_feed)).any(), torch.tensor(True))
        pass

    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        blinear = BayesianLinear(10, 10)
        to_feed = torch.ones((1, 10))

        predicted = blinear(to_feed)
        complexity_cost = blinear.log_variational_posterior - blinear.log_prior

        self.assertEqual((complexity_cost == complexity_cost).all(), torch.tensor(True))
        pass

    def test_inheritance(self):

        #check if bayesian linear has nn.Module and BayesianModule classes
        blinear = BayesianLinear(10, 10)
        self.assertEqual(isinstance(blinear, (nn.Module)), True)
        self.assertEqual(isinstance(blinear, (BayesianModule)), True)

    def test_sequential_cpu(self):
        #check if we can create sequential models chaning our Bayesian Linear layers
        model = nn.Sequential(BayesianLinear(10, 10),
                              nn.Linear(10, 15),
                              BayesianLinear(15,10))

        to_feed = torch.ones((1, 10))
        #if this works, the test will pass
        result = model(to_feed)
        pass
    
    def test_sequential_cuda(self):
        #check if we can create sequential models chaning our Bayesian Linear layers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = nn.Sequential(BayesianLinear(10, 10),
                              nn.Linear(10, 15),
                              BayesianLinear(15,10)).to(device)

        to_feed = torch.ones((1, 10)).to(device)
        #if this works, the test will pass
        result = model(to_feed)
        pass
    
if __name__ == "__main__":
    unittest.main()