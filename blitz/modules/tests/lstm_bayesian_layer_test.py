import unittest
import torch
from torch import nn

from blitz.modules import BayesianLSTM
from blitz.modules.base_bayesian_module import BayesianModule

class TestLinearBayesian(unittest.TestCase):
    def test_init_bayesian_lstm(self):
        b_lstm = BayesianLSTM(10, 10)
        self.assertEqual(isinstance(b_lstm, (nn.Module)), True)
        self.assertEqual(isinstance(b_lstm, (BayesianModule)), True)
        pass
    
    def test_infer_shape_1_sample(self):
        deterministic_lstm = nn.LSTM(1, 10, 1, batch_first=True)
        in_data = torch.ones((10, 10, 1))
        det_inference = deterministic_lstm(in_data)
        
        b_lstm = BayesianLSTM(1, 10)
        b_inference = b_lstm(in_data, hidden_states=None)
        
        self.assertEqual(det_inference[0].shape, b_inference[0].shape)
        pass
    
    def test_variational_inference(self):
        #create module, check if inference is variating
        deterministic_lstm = nn.LSTM(1, 10, 1, batch_first=True)
        in_data = torch.ones((10, 10, 1))
        det_inference = deterministic_lstm(in_data)
        
        b_lstm = BayesianLSTM(1, 10)
        b_inference_1 = b_lstm(in_data, hidden_states=None)
        b_inference_2 = b_lstm(in_data, hidden_states=None)
        
        self.assertEqual((b_inference_1[0] != b_inference_2[0]).any(), torch.tensor(True))
        self.assertEqual((det_inference[0] == det_inference[0]).all(), torch.tensor(True))
        pass
    
    def test_frozen_inference(self):
        b_lstm = BayesianLSTM(1, 10)
        b_lstm.freeze = True
        
        in_data = torch.ones((10, 10, 1))
        b_inference_1 = b_lstm(in_data, hidden_states=None)
        b_inference_2 = b_lstm(in_data, hidden_states=None)
        
        self.assertEqual((b_inference_1[0] == b_inference_2[0]).all(), torch.tensor(True))
    
    def test_peephole_inference(self):
        b_lstm = BayesianLSTM(1, 10, peephole=True)
        in_data = torch.ones((10, 10, 1))
        b_lstm(in_data)
        b_lstm.freeze = True
        b_lstm(in_data)
    
    def test_kl_divergence(self):
        #create model, sample weights
        #check if kl divergence between apriori and a posteriori is working
        b_lstm = BayesianLSTM(1, 10)
        to_feed = torch.ones((10, 10, 1))

        predicted = b_lstm(to_feed)
        complexity_cost = b_lstm.log_variational_posterior - b_lstm.log_prior

        self.assertEqual((complexity_cost == complexity_cost).all(), torch.tensor(True))
    
    def test_sequential_cuda(self):
        #check if we can create sequential models chaning our Bayesian Linear layers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b_lstm = BayesianLSTM(1, 10)
        to_feed = torch.ones((10, 10, 1))

        predicted = b_lstm(to_feed)
        
        
    
if __name__ == "__main__":
    unittest.main()