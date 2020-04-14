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
        deterministic_lstm = nn.LSTM(1, 100, 1, batch_first=True)
        in_data = torch.ones((10, 10, 1))
        det_inference = deterministic_lstm(in_data)
    
if __name__ == "__main__":
    unittest.main()