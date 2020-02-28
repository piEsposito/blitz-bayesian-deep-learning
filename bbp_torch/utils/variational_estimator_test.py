import unittest

class TestVariationalInference(unittest.TestCase):

    def test_monte_carlo_inference(self):
        #create model
        #do two inferences over same datapoint, check if different
        pass

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