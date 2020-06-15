import unittest
import torch
from blitz.modules.weight_sampler import TrainableRandomDistribution, PriorWeightDistribution
from blitz.modules import BayesianLinear

class TestWeightSampler(unittest.TestCase):

    def test_gaussian_sample(self):
        #checks if sample works

        mu = torch.Tensor(10, 10).uniform_(-1, 1)
        rho = torch.Tensor(10, 10).uniform_(-1, 1)

        dist = TrainableRandomDistribution(mu, rho)
        s1 = dist.sample()
        s2 = dist.sample()

        self.assertEqual((s1 != s2).any(), torch.tensor(True))
        self.assertEqual(mu.shape, s1.shape)
        self.assertEqual(rho.shape, s1.shape)
        pass
        
    def test_gaussian_log_posterior(self):
        #checks if it the log_posterior calculator is working

        mu = torch.Tensor(10, 10).uniform_(-1, 1)
        rho = torch.Tensor(10, 10).uniform_(-1, 1)

        dist = TrainableRandomDistribution(mu, rho)
        s1 = dist.sample()

        log_posterior = dist.log_posterior()
        #check if it is not none
        self.assertEqual(log_posterior == log_posterior, torch.tensor(True))

    def test_scale_mixture_prior(self):
        mu = torch.Tensor(10, 10).uniform_(-1, 1)
        rho = torch.Tensor(10, 10).uniform_(-1, 1)

        dist = TrainableRandomDistribution(mu, rho)
        s1 = dist.sample()

        log_posterior = dist.log_posterior()

        prior_dist = PriorWeightDistribution(0.5, 1, .002)
        log_prior = prior_dist.log_prior(s1)

        #print(log_prior)
        #print(log_posterior)
        self.assertEqual(log_prior == log_prior, torch.tensor(True))
        self.assertEqual(log_posterior <= log_posterior - log_prior, torch.tensor(True))
        pass

    def test_scale_mixture_any_prior(self):
        mu = torch.Tensor(10, 10).uniform_(-1, 1)
        rho = torch.Tensor(10, 10).uniform_(-1, 1)

        dist = TrainableRandomDistribution(mu, rho)
        s1 = dist.sample()

        log_posterior = dist.log_posterior()

        prior_dist = PriorWeightDistribution(dist=torch.distributions.studentT.StudentT(1, 1))
        log_prior = prior_dist.log_prior(s1)

        #print(log_prior)
        #print(log_posterior)
        self.assertEqual(log_prior == log_prior, torch.tensor(True))
        self.assertEqual(log_posterior <= log_posterior - log_prior, torch.tensor(True))
        pass

    def test_any_prior_on_layer(self):
        l = BayesianLinear(7, 5, prior_dist=torch.distributions.studentT.StudentT(1, 1))
        t = torch.ones(3, 7)
        _ = l(t)

        self.assertEqual(l.log_prior, l.log_prior)
        pass

if __name__ == "__main__":
    unittest.main()