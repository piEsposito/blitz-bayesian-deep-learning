import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import GaussianVariational, ScaleMixturePrior


class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
    
    """
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

    def forward(self, x):
        # Sample the weights
        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features))
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))
