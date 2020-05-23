# Bayesian Neural Network layers
They all inherit from torch.nn.Module
# Index:
  * [BayesianModule](#class-BayesianModule)
  * [BayesianLinear](#class-BayesianLinear)
  * [BayesianConv1d](#class-BayesianConv1d)
  * [BayesianConv2d](#class-BayesianConv2d)
  * [BayesianConv3d](#class-BayesianConv3d)
  * [BayesianLSTM](#class-BayesianLSTM)
  * [BayesianGRU](#class-BayesianGRU)
  * [BayesianEmbedding](#class-BayesianEmbedding)

---
## class BayesianModule(torch.nn.Module)
### blitz.modules.base_bayesian_module.BayesianModule()
Implements a as-interface used BayesianModule to enable further specific behavior
Inherits from torch.nn.Module

---

## class BayesianLinear
### blitz.modules.BayesianLinear(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)

Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * bias bool ->  wheter the model will have biases
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward():
      
      Performs a feedforward operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded

---
## class BayesianConv1d
### blitz.modules.BayesianConv1d(in_channels, out_channels, kernel_size, groups = 1, stride = 1, padding = 0, dilation = 1, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)
DESCRIPTION

#### Parameters:
  * in_channels int -> incoming channels for the layer
  * out_channels int -> output channels for the layer
  * kernel_size int -> size of the kernels for this convolution layer
  * groups int -> number of groups on which the convolutions will happend
  * padding int -> size of padding (0 if no padding)
  * dilation int -> dilation of the weights applied on the input tensor
  * bias bool -> whether the bias will exist (True) or set to zero (False)
  * prior_sigma_1 float -> prior sigma on the mixture prior distribution 1
  * prior_sigma_2 float -> prior sigma on the mixture prior distribution 2
  * prior_pi float -> pi on the scaled mixture prior
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * freeze bool -> wheter the model will start with frozen(deterministic) weights, or not
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward():
      
      Performs a feedforward Conv3d operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward Conv2d operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded
    

---

## class BayesianConv2d
### blitz.modules.BayesianConv2d(in_channels, out_channels, kernel_size, groups = 1, stride = 1, padding = 0, dilation = 1, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)
DESCRIPTION

#### Parameters:
  * in_channels int -> incoming channels for the layer
  * out_channels int -> output channels for the layer
  * kernel_size tuple (int, int) -> size of the kernels for this convolution layer
  * groups int -> number of groups on which the convolutions will happend
  * padding int -> size of padding (0 if no padding)
  * dilation int -> dilation of the weights applied on the input tensor
  * bias bool -> whether the bias will exist (True) or set to zero (False)
  * prior_sigma_1 float -> prior sigma on the mixture prior distribution 1
  * prior_sigma_2 float -> prior sigma on the mixture prior distribution 2
  * prior_pi float -> pi on the scaled mixture prior
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * freeze bool -> wheter the model will start with frozen(deterministic) weights, or not
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward():
      
      Performs a feedforward Conv2d operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward Conv2d operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded
    
---

## class BayesianConv3d
### blitz.modules.BayesianConv2d(in_channels, out_channels, kernel_size, groups = 1, stride = 1, padding = 0, dilation = 1, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)
DESCRIPTION

#### Parameters:
  * in_channels int -> incoming channels for the layer
  * out_channels int -> output channels for the layer
  * kernel_size tuple (int, int, int) -> size of the kernels for this convolution layer
  * groups int -> number of groups on which the convolutions will happend
  * padding int -> size of padding (0 if no padding)
  * dilation int -> dilation of the weights applied on the input tensor
  * bias bool -> whether the bias will exist (True) or set to zero (False)
  * prior_sigma_1 float -> prior sigma on the mixture prior distribution 1
  * prior_sigma_2 float -> prior sigma on the mixture prior distribution 2
  * prior_pi float -> pi on the scaled mixture prior
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * freeze bool -> wheter the model will start with frozen(deterministic) weights, or not
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward():
      
      Performs a feedforward Conv3d operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
      
   * forward_frozen(x):
      
      Performs a feedforward Conv2d operation using onle the mu tensor as weights. 
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x = torch.tensor corresponding to the datapoints tensor to be feedforwarded
    
---

## class BayesianLSTM
### blitz.modules.BayesianLSTM(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False, peephole = False)

Bayesian LSTM layer, implements the LSTM layer using the weight uncertainty tools proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * bias bool ->  wheter the model will have biases
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  * peephole bool -> if the lstm shoudl use peephole connections rather than default ones
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward(x, ):
      
      Performs a feedforward operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns  tuple of format (torch.tensor, (torch.tensor, torch.tensor)), representing the output and hidden state of the LSTM layer
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
       * hidden_states - None or tupl of the format (torch.tensor, torch.tensor), representing the hidden states of the network. Internally, if None, consider zeros of the proper format).
      
   * sample_weights():
      
      Assings internally its weights to be used on feedforward operations by sampling it from its GaussianVariational
      
   * get_frozen_weights():
   
      Assings internally for its weights deterministaclly the mean of its GaussianVariational sampler.
      
---

## class BayesianGRU
### blitz.modules.BayesianGRU(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)

Bayesian GRU layer, implements the GRU layer using the weight uncertainty tools proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * bias bool ->  wheter the model will have biases
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward(x, ):
      
      Performs a feedforward operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns  tuple of format (torch.tensor, (torch.tensor, torch.tensor)), representing the output and hidden state of the GRU layer
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
       * hidden_states - None or tupl of the format (torch.tensor, torch.tensor), representing the hidden states of the network. Internally, if None, consider zeros of the proper format).
      
   * sample_weights():
      
      Assings internally its weights to be used on feedforward operations by sampling it from its GaussianVariational
      
   * get_frozen_weights():
   
      Assings internally for its weights deterministaclly the mean of its GaussianVariational sampler.
      
---

## class BayesianEmbedding
### blitz.modules.BayesianEmbedding (num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False,)

Bayesian Embedding layer, implements the Embedding layer using the weight uncertainty tools proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). 

Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.

Inherits from BayesianModule

#### Parameters:
  * num_embedding int -> Size of the vocabulary
  * embedding_dim int -> Dimension of the embedding
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
  * padding_idx int -> If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index
  * max_norm float -> If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
  * norm_type float -> The p of the p-norm to compute for the max_norm option. Default 2.
  * scale_grad_by_freq -> If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
  * sparse bool -> If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
  
  
#### Methods:
  * forward(x, ):
      
      Performs a embedding operation with sampled weights. If the model is frozen uses only the expected values.
      
      Returns  tuple of format (torch.tensor, (torch.tensor, torch.tensor)), representing the output and hidden state of the GRU layer
      
      Description
      ##### Parameters
       * x - torch.tensor corresponding to the datapoints tensor to be feedforwarded
       * hidden_states - None or tupl of the format (torch.tensor, torch.tensor), representing the hidden states of the network. Internally, if None, consider zeros of the proper format).
      
   * sample_weights():
      
      Assings internally its weights to be used on feedforward operations by sampling it from its GaussianVariational
      
    
