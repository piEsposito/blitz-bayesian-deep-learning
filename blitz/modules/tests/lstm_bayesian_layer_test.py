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
        
    
if __name__ == "__main__":
    unittest.main()