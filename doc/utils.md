# Utils and decorators to enable easy basyesian training and inference

# Index:
  * [Decorator variational_estimator](#Variational-Estimator)
---
## Variational Estimator

Dynamically adds some util methods to object that inherits from torch.nn.Module in order to facilitate bayesian training and inference.

### @variational_estimator(model)
  #### Parameters:
  * model: -> torch.nn.Module to have introduced the Bayesian DL methods
    
### Methods introduced:
  * #### nn_kl_divergence()
    
    Returns torch.tensor corresponding to the summed KL divergence (relative to the curren weight sampling) of all of its BayesianModule layers.
    
  * #### freeze_model()
    
    Freezes the model weights by making its BayesianModule layers forward operation use, while not unfrozen, only its weight distribution mean tensor.
    
  * #### unfreeze_model()
  
    Unfreezes the model by letting it sample its weights using the Bayes By Backprop paper proposed algorithm rather than using only its expected value.
    
  * #### sample_elbo(inputs, labels, criterion, sample_nbr)
    
    Samples the ELBO loss of the model sample_nbr times by doing feedforward operations and summing its model kl divergence with the loss the criterion outputs.
    
    ##### Parameters:
      * inputs: torch.tensor -> the input data to the model
      * labels: torch.tensor -> label data for the performance-part of the loss calculation
      
        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
               
      * criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather the performance cost for the model
      * sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to gather the loss to be .backwarded in the optimization of the model.

    #### Returns:
      * loss: torch.tensor -> elbo loss for the data given

  * #### mfvi_forward(inputs, sample_nbr)

  Performs mean-field variational inference for the variational estimator model on the inputs

  #### Parameters:
    * inputs: torch.tensor -> the input data to the model
    * sample_nbr: int -> number of forward passes to be done on the data
  #### Returns:
    * mean_: torch.tensor -> mean of the perdictions along each of the features of each datapoint on the batch axis, for each feature of ea
    * std_: torch.tensor -> std of the predictions along each of the features of each datapoint on the batch axis
  
