import unittest

class TestConv2DBayesian(unittest.TestCase):

    def test_init_bayesian_layer(self):
        #create bayesian layer
        pass

    def test_weights_shape(self):
        #check if weights shape is the expected
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