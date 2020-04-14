import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import GaussianVariational, ScaleMixturePrior


class BayesianLSTM(BayesianModule):
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 prior_sigma_1 = 1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 0.5,
                 freeze = False):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Variational weight parameters and sample for weight ih
        self.weight_ih_mu = nn.Parameter(torch.Tensor(out_features, in_features * 4).uniform_(-0.2, 0.2))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(out_features, in_features * 4).uniform_(-5, -4))
        self.weight_ih_sampler = GaussianVariational(self.weight_ih_mu, self.weight_ih_rho)
        
        # Variational weight parameters and sample for weight hh
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, in_features * 4).uniform_(-0.2, 0.2))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, in_features * 4).uniform_(-5, -4))
        self.weight_hh_sampler = GaussianVariational(self.weight_hh_mu, self.weight_hh_rho)
        
        # Variational weight parameters and sample for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).uniform_(-5, -4))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        
        #our prior distributions
        self.weight_ih_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.weight_hh_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        
        self.log_prior = 0
        self.log_variational_posterior = 0
        
    def forward(self,
                x,
                hidden_states):
        pass