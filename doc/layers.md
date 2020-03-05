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

Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks (Bayes by Backprop paper).
Inherits from BayesianModule

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
