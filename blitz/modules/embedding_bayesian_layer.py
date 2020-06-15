import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import TrainableRandomDistribution, ScaleMixturePrior

class BayesianEmbedding(BayesianModule):
    """
    Bayesian Embedding layer, implements the embedding layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        num_embedding int -> Size of the vocabulary
        embedding_dim int -> Dimension of the embedding
        prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
        prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
        prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
        freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
        padding_idx int -> If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index
        max_norm float -> If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
        norm_type float -> The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq -> If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
        sparse bool -> If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init

    
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        self.freeze = freeze

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it
        
        #if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior()
        self.log_prior = self.weight_prior_dist.log_prior(w)

        return F.embedding(x,
                           w,
                           self.padding_idx,
                           self.max_norm,
                           self.norm_type,
                           self.scale_grad_by_freq,
                           self.sparse)

    def forward_frozen(self, x):
        return F.embedding(x,
                           self.weight_mu,
                           self.padding_idx,
                           self.max_norm,
                           self.norm_type,
                           self.scale_grad_by_freq,
                           self.sparse)