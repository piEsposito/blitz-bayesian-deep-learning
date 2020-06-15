import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

class TrainableRandomDistribution(nn.Module):
    #Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    #Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi
        #self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean

        sets those weights as the current ones

        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self, w=None):

        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """

        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w
        
        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors =  -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2)/(2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()

class ScaleMixturePrior(nn.Module):
    #Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None):
        super().__init__()


        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

        

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))

        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        if self.dist2 is None:
            prob_n2 = 0

        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2)

        return (torch.log(prior_pdf) - 0.5).sum()
