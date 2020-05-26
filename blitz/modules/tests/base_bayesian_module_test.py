import unittest
import torch
from torch import nn

from blitz.modules.base_bayesian_module import BayesianModule, BayesianRNN
from blitz.modules import BayesianLSTM, BayesianGRU

class TestLinearBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        b_module = BayesianModule()
        self.assertEqual(isinstance(b_module, (nn.Module)), True)
        
    def test_init_brnn(self):
        b_module = BayesianRNN()
        self.assertEqual(isinstance(b_module, (nn.Module)), True)
        
    def test_brnn_sharpen_posterior_lstm(self):
        b_module = BayesianLSTM(3, 5, sharpen=True)
        in_tensor = torch.ones(5, 4, 3)
        out_tensor = b_module(in_tensor)[0][:, -1, :]
        
        loss = nn.MSELoss()(out_tensor.clone().detach().normal_(), out_tensor)
        b_module.sharpen_posterior(loss, in_tensor.shape)
    
    def test_brnn_sharpen_posterior_on_forward_lstm(self):
        b_module = BayesianLSTM(3, 5, sharpen=True)
        in_tensor = torch.ones(5, 4, 3)
        out_tensor = b_module(in_tensor)[0][:, -1, :]
        
        loss = nn.MSELoss()(out_tensor.clone().detach().normal_(), out_tensor)
        b_module.forward(in_tensor, sharpen_loss=loss)
        
    def test_brnn_sharpen_posterior_gru(self):
        b_module = BayesianGRU(3, 5, sharpen=True)
        in_tensor = torch.ones(5, 4, 3)
        out_tensor = b_module(in_tensor)[0][:, -1, :]
        
        loss = nn.MSELoss()(out_tensor.clone().detach().normal_(), out_tensor)
        b_module.sharpen_posterior(loss, in_tensor.shape)
    
    def test_brnn_sharpen_posterior_on_forward_gru(self):
        b_module = BayesianGRU(3, 5, sharpen=True)
        in_tensor = torch.ones(5, 4, 3)
        out_tensor = b_module(in_tensor)[0][:, -1, :]
        
        loss = nn.MSELoss()(out_tensor.clone().detach().normal_(), out_tensor)
        b_module.forward(in_tensor, sharpen_loss=loss)
        
if __name__ == "__main__":
    unittest.main()