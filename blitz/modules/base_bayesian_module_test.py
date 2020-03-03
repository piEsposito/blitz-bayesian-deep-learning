import unittest
import torch
from torch import nn

from blitz.modules.base_bayesian_module import BayesianModule

class TestLinearBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        b_module = BayesianModule()
        self.assertEqual(isinstance(b_module, (nn.Module)), True)


if __name__ == "__main__":
    unittest.main()