# Bayesian Neural Network layers
They all inherit from torch.nn.Module
# Index:
  * [BayesianModule](#class-BayesianModule)
  * [BayesianLinear](#class-BayesianLinear)
  * [BayesianConv2d](#class-BayesianConv2d)

---
## class BayesianModule(torch.nn.Module)
### blitz.modules.base_bayesian_module.BayesianModule()
Implements a as-interface used BayesianModule to enable further specific behavior
Inherits from torch.nn.Module

---

## class BayesianLinear
### blitz.modules.BayesianLinear(in_features, out_features, bias=True, prior_sigma_1 = 1, prior_sigma_2 = 0.002, prior_pi = 0.5, freeze = False)

Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper). Creates weight samplers of the class GaussianVariational for the weights and biases to be used on its feedforward ops.
Inherits from BayesianModule

#### Parameters:
  * in_features int - Number nodes of the information to be feedforwarded
  * out_features int - Number of out nodes of the layer
  * bias bool -  wheter the model will have biases
  * prior_sigma_1 float - sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float - sigma of one of the prior w distributions to mixture
  * prior_pi float - factor to scale the gaussian mixture of the model prior distribution
  * freeze - wheter the model is frozen (will use deterministic weights on the feedforward op)
  
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
## class BayesianConv2d
### blitz.modules.BayesianConv2d
DESCRIPTION

#### Parameters:
  * p1 - description
  * p2 - description 
  
#### Methods:
  * m1():
  
      Description
      
   * m1():
  
      Description
      
   * m1():
  
      Description
    
---
