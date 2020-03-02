import unittest
import torch
from torch import nn
from torch.nn import functional as F

from bbp_torch.modules.base_bayesian_module import BayesianModule
from bbp_torch.modules.weight_sampler import GaussianVariational, ScaleMixturePrior

class BayesianConv2d(BayesianModule):

    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups = 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 bias=True,
                 prior_sigma_1 = 1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 0.5):
        super().__init__()
        
        #our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        #our weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).uniform_(-5, -4))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)

        #our biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features))
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv2d(input=x,
                        weight=w,
                        bias=b,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)