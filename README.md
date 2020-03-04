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

As we know, on deterministic (non bayesian) neural network layers, the trainable parameters correspond directly to the weights used on its linear transformation of the previous one (or the input, if it is the case). It corresponds to the following equation. (Z correspond to the activated-output of the layer i:


![equation](https://latex.codecogs.com/gif.latex?a^{(i&plus;1)}&space;=&space;W^{(i&plus;1)}\cdot&space;z^{(i)}&space;&plus;&space;b^{(i&plus;1)})

## A simple example for regression

We will now see how can Bayesian Deep Learning be used for regression in order to gather confidence interval over our datapoint rather than a prediction, and how this information may be more useful than a low-error predicion.
 
 
###### Made by Pi Esposito
