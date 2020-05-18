import unittest
from blitz.modules import BayesianConv2d, BayesianConv1d, BayesianConv3d
from blitz.modules.base_bayesian_module import BayesianModule

import torch
from torch import nn

class TestConv1DBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        #create bayesian layer

        module = BayesianConv1d(3, 10, 3)
        pass

    def test_weights_shape(self):
       #check if weights shape is the expected
       bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=3)

       conv = nn.Conv1d(in_channels=3,
                        out_channels=3,
                        kernel_size=3)

       to_feed = torch.ones((1, 3, 25))

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_weights_shape_cuda(self):
       #check if weights shape is the expected
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=3).to(device)

       conv = nn.Conv1d(in_channels=3,
                        out_channels=3,
                        kernel_size=3).to(device)

       to_feed = torch.ones((1, 3, 25)).to(device)

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_variational_inference(self):
        #create module, check if inference is variating
        bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=1)

        conv = nn.Conv1d(in_channels=3,
                        out_channels=3,
                        kernel_size=3)

        to_feed = torch.ones((1, 3, 25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))
        self.assertEqual((conv(to_feed) == conv(to_feed)).all(), torch.tensor(True))
        pass


    def test_freeze_module(self):
        #create module, freeze
        #check if two inferences keep equal
        bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=3,
                              bias=False)

        to_feed = torch.ones((1, 3, 25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))

        frozen_feedforward = bconv.forward_frozen(to_feed)
        bconv.freeze = True
        self.assertEqual((bconv.forward(to_feed) == frozen_feedforward).all(), torch.tensor(True))
        pass
    
    def test_inheritance(self):

        #check if bayesian linear has nn.Module and BayesianModule classes
        bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=3,
                              bias=False)

        self.assertEqual(isinstance(bconv, (nn.Module)), True)
        self.assertEqual(isinstance(bconv, (BayesianModule)), True)

    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        bconv = BayesianConv1d(in_channels=3,
                              out_channels=3,
                              kernel_size=3)

        to_feed = torch.ones((1, 3, 25))
        predicted = bconv(to_feed)
        
        complexity_cost = bconv.log_variational_posterior - bconv.log_prior
        self.assertEqual((complexity_cost == complexity_cost).all(), torch.tensor(True))
        pass

class TestConv2DBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        #create bayesian layer

        module = BayesianConv2d(3, 10, (3,3))
        pass

    def test_weights_shape(self):
       #check if weights shape is the expected
       bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3))

       conv = nn.Conv2d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3))

       to_feed = torch.ones((1, 3, 25, 25))

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_weights_shape_cuda(self):
       #check if weights shape is the expected
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3)).to(device)

       conv = nn.Conv2d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3)).to(device)

       to_feed = torch.ones((1, 3, 25, 25)).to(device)

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_variational_inference(self):
        #create module, check if inference is variating
        bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3))

        conv = nn.Conv2d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3))

        to_feed = torch.ones((1, 3, 25, 25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))
        self.assertEqual((conv(to_feed) == conv(to_feed)).all(), torch.tensor(True))
        pass


    def test_freeze_module(self):
        #create module, freeze
        #check if two inferences keep equal
        bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3),
                              bias=False)

        to_feed = torch.ones((1, 3, 25, 25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))

        frozen_feedforward = bconv.forward_frozen(to_feed)
        bconv.freeze = True
        self.assertEqual((bconv.forward(to_feed) == frozen_feedforward).all(), torch.tensor(True))
        pass
    
    def test_inheritance(self):

        #check if bayesian linear has nn.Module and BayesianModule classes
        bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3),
                              bias=False)

        self.assertEqual(isinstance(bconv, (nn.Module)), True)
        self.assertEqual(isinstance(bconv, (BayesianModule)), True)

    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3))

        to_feed = torch.ones((1, 3, 25, 25))
        predicted = bconv(to_feed)
        
        complexity_cost = bconv.log_variational_posterior - bconv.log_prior
        self.assertEqual((complexity_cost == complexity_cost).all(), torch.tensor(True))
        pass
class TestConv3DBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        #create bayesian layer

        module = BayesianConv3d(3, 10, (3,3, 3))
        pass

    def test_weights_shape(self):
       #check if weights shape is the expected
       bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3))

       conv = nn.Conv3d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3,3))

       to_feed = torch.ones((1, 3, 25, 25, 25))

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_weights_shape_cuda(self):
       #check if weights shape is the expected
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3)).to(device)

       conv = nn.Conv3d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3,3)).to(device)

       to_feed = torch.ones((1, 3, 25, 25, 25)).to(device)

       infer1 = bconv(to_feed)
       infer2 = conv(to_feed)
       
       self.assertEqual(infer1.shape, infer2.shape)
       pass

    def test_variational_inference(self):
        #create module, check if inference is variating
        bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3))

        conv = nn.Conv3d(in_channels=3,
                        out_channels=3,
                        kernel_size=(3,3,3))

        to_feed = torch.ones((1, 3, 25, 25,25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))
        self.assertEqual((conv(to_feed) == conv(to_feed)).all(), torch.tensor(True))
        pass


    def test_freeze_module(self):
        #create module, freeze
        #check if two inferences keep equal
        bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3),
                              bias=False)

        to_feed = torch.ones((1, 3, 25, 25,25))

        self.assertEqual((bconv(to_feed) != bconv(to_feed)).any(), torch.tensor(True))

        frozen_feedforward = bconv.forward_frozen(to_feed)
        bconv.freeze = True
        self.assertEqual((bconv.forward(to_feed) == frozen_feedforward).all(), torch.tensor(True))
        pass
    
    def test_inheritance(self):

        #check if bayesian linear has nn.Module and BayesianModule classes
        bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3),
                              bias=False)

        self.assertEqual(isinstance(bconv, (nn.Module)), True)
        self.assertEqual(isinstance(bconv, (BayesianModule)), True)

    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        bconv = BayesianConv3d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3,3))

        to_feed = torch.ones((1, 3, 25, 25,25))
        predicted = bconv(to_feed)
        
        complexity_cost = bconv.log_variational_posterior - bconv.log_prior
        self.assertEqual((complexity_cost == complexity_cost).all(), torch.tensor(True))
        pass

if __name__ == "__main__":
    unittest.main()