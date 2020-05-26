import torch
from torch import nn
import math

class BayesianModule(nn.Module):
    """
    creates base class for BNN, in order to enable specific behavior
    """
    def init(self):
        super().__init__()
        
        
class BayesianRNN(BayesianModule):
    """
    implements base class for B-RNN to enable posterior sharpening
    """
    def __init__(self,
                 sharpen=False):
        super().__init__()
        
        self.weight_ih_mu = None
        self.weight_hh_mu = None
        self.bias = None
        
        self.weight_ih_sampler = None
        self.weight_hh_sampler = None
        self.bias_sampler = None

        self.weight_ih = None
        self.weight_hh = None
        self.bias = None
        
        self.sharpen = sharpen
        
        self.weight_ih_eta = None
        self.weight_hh_eta = None
        self.bias_eta = None
        self.ff_parameters = None
        self.loss_to_sharpen = None
        
    
    def sample_weights(self):
        pass
    
    def init_sharpen_parameters(self):
        if self.sharpen:
            self.weight_ih_eta = nn.Parameter(torch.Tensor(self.weight_ih_mu.size()))
            self.weight_hh_eta = nn.Parameter(torch.Tensor(self.weight_hh_mu.size()))
            self.bias_eta = nn.Parameter(torch.Tensor(self.bias_mu.size()))
            
            self.ff_parameters = []

            self.init_eta()
    
    def init_eta(self):
        stdv = 1.0 / math.sqrt(self.weight_hh_eta.shape[0]) #correspond to hidden_units parameter
        self.weight_ih_eta.data.uniform_(-stdv, stdv)
        self.weight_hh_eta.data.uniform_(-stdv, stdv)
        self.bias_eta.data.uniform_(-stdv, stdv)
    
    def set_loss_to_sharpen(self, loss):
        self.loss_to_sharpen = loss
    
    def sharpen_posterior(self, loss, input_shape):
        """
        sharpens the posterior distribution by using the algorithm proposed in
        @article{DBLP:journals/corr/FortunatoBV17,
          author    = {Meire Fortunato and
                       Charles Blundell and
                       Oriol Vinyals},
          title     = {Bayesian Recurrent Neural Networks},
          journal   = {CoRR},
          volume    = {abs/1704.02798},
          year      = {2017},
          url       = {http://arxiv.org/abs/1704.02798},
          archivePrefix = {arXiv},
          eprint    = {1704.02798},
          timestamp = {Mon, 13 Aug 2018 16:48:21 +0200},
          biburl    = {https://dblp.org/rec/journals/corr/FortunatoBV17.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }
        """
        bs, seq_len, in_size = input_shape
        gradients = torch.autograd.grad(outputs=loss,
                                        inputs=self.ff_parameters,
                                        grad_outputs=torch.ones(loss.size()).to(loss.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)
        
        grad_weight_ih, grad_weight_hh, grad_bias = gradients
        
        #to generate sigmas on the weight sampler
        _ = self.sample_weights()
        
        weight_ih_sharpened = self.weight_ih_mu - self.weight_ih_eta * grad_weight_ih + self.weight_ih_sampler.sigma
        weight_hh_sharpened = self.weight_hh_mu - self.weight_hh_eta * grad_weight_hh + self.weight_hh_sampler.sigma
        bias_sharpened = self.bias_mu - self.bias_eta * grad_bias + self.bias_sampler.sigma
        
        if self.bias is not None:
            b_log_posterior = self.bias_sampler.log_posterior(w=bias_sharpened)
            b_log_prior_ = self.bias_prior_dist.log_prior(bias_sharpened)
            
        else:
            b_log_posterior = b_log_prior = 0
        
        
        self.log_variational_posterior += (self.weight_ih_sampler.log_posterior(w=weight_ih_sharpened) + b_log_posterior + self.weight_hh_sampler.log_posterior(w=weight_hh_sharpened)) / seq_len
        
        self.log_prior += self.weight_ih_prior_dist.log_prior(weight_ih_sharpened) + b_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh_sharpened) / seq_len
        
        return weight_ih_sharpened, weight_hh_sharpened, bias_sharpened

        