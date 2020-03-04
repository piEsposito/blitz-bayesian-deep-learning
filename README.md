# Blitz - Bayesian Layers in Torch Zoo
BLiTZ is a simple and extensible library to create Bayesian Neural Network Layers (based on whats proposed in [Weight Uncertainty in Neural Networks paper](https://arxiv.org/abs/1505.05424)) on PyTorch. By using BLiTZ layers and utils, you can add uncertanity and gather the complexity cost of your model in a simple way that does not affect the interaction between your layers, as if you were using standard PyTorch.

By using our core weight sampler classes, you can extend and improve this library to add uncertanity to a bigger scope of layers as you will in a well-integrated to PyTorch way. Also pull requests are welcome.

Our objective is empower people to apply Bayesian Deep Learning by focusing rather on their idea, and not the hard-coding part. 

## Install


To install it, just git-clone it and pip-install it locally:

```
git clone https://github.com/piEsposito/blitz-bayesian-deep-learning.git
cd blitz
pip install .
```

Later on, we will submit it to **Pypi**.

## Documentation

Documentation for our layers, weight (and prior distribution) sampler and utils:
 * [Bayesian Layers](doc/layers.md)
 * [Weight and prior distribution samplers](doc/samplers.md)
 * [Utils (for easy integration with PyTorch)](doc/utils.md)
 * [Losses](doc/losses.md)
 
## Bayesian Deep Learning in a Nutshell
A very fast explanation of how is uncertainity introduced in Bayesian Neural Networks and how we model its loss in order to objectively improve the confidence over its prediction and reduce the variance without dropout. 

### First of all, a deterministic NN layer linear-transformation

As we know, on deterministic (non bayesian) neural network layers, the trainable parameters correspond directly to the weights used on its linear transformation of the previous one (or the input, if it is the case). It corresponds to the following equation:


![equation](https://latex.codecogs.com/gif.latex?a^{(i&plus;1)}&space;=&space;W^{(i&plus;1)}\cdot&space;z^{(i)}&space;&plus;&space;b^{(i&plus;1)}) 

*(Z correspond to the activated-output of the layer i)*

### The purpose of Bayesian Layers

Bayesian layers seek to introduce uncertainity on its weights by sampling them from a distribution parametrized by trainable variables on each feedforward operation. 

This allows we not just to optimize the performance metrics of the model, but also gather the uncertainity of the network predictions over a specific datapoint (by sampling it much times and measuring the dispersion) and aimingly reduce as much as possible the variance of the network over the prediction, making possible to know how much of incertainity we still have over the label if we try to model it in function of our specific datapoint.

### Weight sampling on Bayesian Layers
To do so, on each feedforward operation we sample the parameters of the linear transformation with the following equations (where **ρ** parametrizes the standard deviation and **μ** parametrizes the mean for the samples linear transformation parameters) :

For the weights:

![equation](https://latex.codecogs.com/gif.latex?W^{(i)}_{(n)}&space;=&space;\mathcal{N}(0,1)&space;*&space;log(1&space;&plus;&space;\rho^{(i)}&space;)&space;&plus;&space;\mu^{(i)})

*Where the sampled W corresponds to the weights used on the linear transformation for the ith layer on the nth sample.*

For the biases:

![equation](https://latex.codecogs.com/gif.latex?b^{(i)}_{(n)}&space;=&space;\mathcal{N}(0,1)&space;*&space;log(1&space;&plus;&space;\rho^{(i)}&space;)&space;&plus;&space;\mu^{(i)})

*Where the sampled b corresponds to the biases used on the linear transformation for the ith layer on the nth sample.*

### It is possible to optimize our trainable weights:

Even tough we have a random multiplier for our weights and biases, it is possible to optimize them by, given some differentiable function of the weights sampled and trainable parameters (in our case, the loss), summing the derivative of the function relative to both of them:

1. Let ![equation](https://latex.codecogs.com/gif.latex?\epsilon&space;=&space;\mathcal{N}(0,1))
2. Let ![equation](https://latex.codecogs.com/gif.latex?\theta&space;=&space;(\rho,&space;\mu))
3. Let ![equation](https://latex.codecogs.com/gif.latex?f(w,&space;\theta)) be differentiable relative to its variables

Therefore:

4. ![equation](https://latex.codecogs.com/gif.latex?\Delta_{\mu}&space;=&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;w}&space;&plus;&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;\mu})

and


5. ![equation](https://latex.codecogs.com/gif.latex?\Delta_{\rho}&space;=&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;w}&space;\frac{\epsilon}{1&space;&plus;&space;e^\rho&space;}&space;&plus;&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;\rho})


## A simple example for regression

We will now see how can Bayesian Deep Learning be used for regression in order to gather confidence interval over our datapoint rather than a prediction, and how this information may be more useful than a low-error predicion.
 
 
###### Made by Pi Esposito
