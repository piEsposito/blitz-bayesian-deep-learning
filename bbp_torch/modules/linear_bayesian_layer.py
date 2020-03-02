import torch
from torch import nn
from torch.nn import functional as F

from weight_sampler import GaussianVariational, ScaleMixturePrior


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 0.5):
        super().__init__()

        #our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0