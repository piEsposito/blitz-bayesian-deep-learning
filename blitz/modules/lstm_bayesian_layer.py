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
        self.use_bias = bias
        self.freeze = freeze
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Variational weight parameters and sample for weight ih
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).uniform_(-0.2, 0.2))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).uniform_(-5, -4))
        self.weight_ih_sampler = GaussianVariational(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        
        # Variational weight parameters and sample for weight hh
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).uniform_(-0.2, 0.2))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).uniform_(-5, -4))
        self.weight_hh_sampler = GaussianVariational(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        
        # Variational weight parameters and sample for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).uniform_(-5, -4))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.bias=None
        
        #our prior distributions
        self.weight_ih_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.weight_hh_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    
    def sample_weights(self):
        #sample weights
        self.weight_ih = self.weight_ih_sampler.sample()
        self.weight_hh = self.weight_hh_sampler.sample()
        
        #if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
            
        else:
            b = torch.zeros((self.out_features * 4))
            b_log_posterior = 0
            b_log_prior = 0
            
        self.bias = b
        
        #gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_log_posterior + self.weight_ih_sampler.log_posterior()
        
        self.log_prior = self.weight_ih_prior_dist.log_prior(self.weight_ih) + b_log_prior + self.weight_hh_prior_dist.log_prior(self.weight_hh)
        
            
    
    def forward(self,
                x,
                hidden_states=None):
        
        #Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            h_t, c_t = (torch.zeros(self.out_features).to(x.device), 
                        torch.zeros(self.out_features).to(x.device))
        else:
            h_t, c_t = hidden_states
            
        self.sample_weights()
        
        #simplifying our out features, and hidden seq list
        HS = self.out_features
        hidden_seq = []
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)
        