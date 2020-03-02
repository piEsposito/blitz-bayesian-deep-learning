import unittest
from bbp_torch.modules import BayesianConv2d

import torch
from torch import nn

class TestConv2DBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        #create bayesian layer

        module = BayesianConv2d(3, 10, (3,3))
        pass

    def test_weights_shape(self):
       #check if weights shape is the expected
       to_feed = torch.ones((1, 3, 25, 25))
       bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3))

       conv = nn.Conv2d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3))

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_variational_inference(self):
        #create module, check if inference is variating
        pass

    def test_freeze_module(self):
        #create module, freeze
        #check if two inferences keep equal
        pass

    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        pass

if __name__ == "__main__":
    unittest.main()