import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import GaussianVariational, ScaleMixturePrior


class BayesianLSTM(BayesianModule):
    """
    Bayesian LSTM layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None,
                 peephole = False):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze
        self.peephole = peephole
        
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        
        # Variational weight parameters and sample for weight ih
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = GaussianVariational(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        
        # Variational weight parameters and sample for weight hh
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = GaussianVariational(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        
        # Variational weight parameters and sample for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.bias=None
        
        #our prior distributions
        self.weight_ih_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.weight_hh_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    
    def sample_weights(self):
        #sample weights
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        
        #if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
            
        else:
            b = 0
            b_log_posterior = 0
            b_log_prior = 0
            
        bias = b
        
        #gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_log_posterior + self.weight_ih_sampler.log_posterior()
        
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh)
        return weight_ih, weight_hh, bias
        
    def get_frozen_weights(self):
        
        #get all deterministic weights
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias = self.bias_mu
        else:
            bias = 0

        return weight_ih, weight_hh, bias

    
    def forward_(self,
                 x,
                 hidden_states):
        
        weight_ih, weight_hh, bias = self.sample_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            h_t, c_t = (torch.zeros(bs, self.out_features).to(x.device), 
                        torch.zeros(bs, self.out_features).to(x.device))
        else:
            h_t, c_t = hidden_states
        
        #simplifying our out features, and hidden seq list
        HS = self.out_features
        hidden_seq = []
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            
            if self.peephole:
                gates = x_t @ weight_ih + c_t @ weight_hh + bias
            else:
                gates = x_t @ weight_ih + h_t @ weight_hh + bias
                g_t = torch.tanh(gates[:, HS*2:HS*3])
            
            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ weight_ih + bias)[:, HS*2:HS*3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                
            hidden_seq.append(h_t.unsqueeze(0))
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)

    def forward_frozen(self,
                       x,
                       hidden_states):

        weight_ih, weight_hh, bias = self.get_frozen_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            h_t, c_t = (torch.zeros(bs, self.out_features).to(x.device), 
                        torch.zeros(bs, self.out_features).to(x.device))
        else:
            h_t, c_t = hidden_states
        
        #simplifying our out features, and hidden seq list
        HS = self.out_features
        hidden_seq = []
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            
            if self.peephole:
                gates = x_t @ weight_ih + c_t @ weight_hh + bias
            else:
                gates = x_t @ weight_ih + h_t @ weight_hh + bias
                g_t = torch.tanh(gates[:, HS*2:HS*3])
            
            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ weight_ih + bias)[:, HS*2:HS*3]
                h_t = torch.sigmoid(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                
            hidden_seq.append(h_t.unsqueeze(0))
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)

    def forward(self,
                x,
                hidden_states=None):

        if self.freeze:
            return self.forward_frozen(x, hidden_states)

        return self.forward_(x, hidden_states)   
        
