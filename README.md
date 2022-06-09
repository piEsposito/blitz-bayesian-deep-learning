# Blitz - Bayesian Layers in Torch Zoo

[![Downloads](https://pepy.tech/badge/blitz-bayesian-pytorch)](https://pepy.tech/project/blitz-bayesian-pytorch)

BLiTZ is a simple and extensible library to create Bayesian Neural Network Layers (based on whats proposed in [Weight Uncertainty in Neural Networks paper](https://arxiv.org/abs/1505.05424)) on PyTorch. By using BLiTZ layers and utils, you can add uncertanity and gather the complexity cost of your model in a simple way that does not affect the interaction between your layers, as if you were using standard PyTorch.

By using our core weight sampler classes, you can extend and improve this library to add uncertanity to a bigger scope of layers as you will in a well-integrated to PyTorch way. Also pull requests are welcome.

 
# Index
 * [Install](#Install)
 * [Documentation](#Documentation)
 * [A simple example for regression](#A-simple-example-for-regression)
   * [Importing the necessary modules](#Importing-the-necessary-modules)
   * [Loading and scaling data](#Loading-and-scaling-data)
   * [Creating our variational regressor class](#Creating-our-variational-regressor-class)
   * [Defining a confidence interval evaluating function](#Defining-a-confidence-interval-evaluating-function)
   * [Creating our regressor and loading data](#Creating-our-regressor-and-loading-data)
   * [Our main training and evaluating loop](#Our-main-training-and-evaluating-loop)
 * [Bayesian Deep Learning in a Nutshell](#Bayesian-Deep-Learning-in-a-Nutshell)
   * [First of all, a deterministic NN layer linear-transformation](#First-of-all,-a-deterministic-NN-layer-linear-transformation)
   * [The purpose of Bayesian Layers](#The-purpose-of-Bayesian-Layers)
   * [Weight sampling on Bayesian Layers](#Weight-sampling-on-Bayesian-Layers)
   * [It is possible to optimize our trainable weights](#It-is-possible-to-optimize-our-trainable-weights)
   * [It is also true that there is complexity cost function differentiable along its variables](#It-is-also-true-that-there-is-complexity-cost-function-differentiable-along-its-variables)
   * [To get the whole cost function at the nth sample](#To-get-the-whole-cost-function-at-the-nth-sample)
   * [Some notes and wrap up](#Some-notes-and-wrap-up)
 * [Citing](#Citing)
 * [References](#References)
   
   
## Install

To install BLiTZ you can use pip command:

```
pip install blitz-bayesian-pytorch
```
Or, via conda:

```
conda install -c conda-forge blitz-bayesian-pytorch
```

You can also git-clone it and pip-install it locally:

```
conda create -n blitz python=3.9
conda activate blitz
git clone https://github.com/piEsposito/blitz-bayesian-deep-learning.git
cd blitz-bayesian-deep-learning
pip install .
```

## Documentation

Documentation for our layers, weight (and prior distribution) sampler and utils:
 * [Bayesian Layers](doc/layers.md)
 * [Weight and prior distribution samplers](doc/samplers.md)
 * [Utils (for easy integration with PyTorch)](doc/utils.md)
 * [Losses](doc/losses.md)

## A simple example for regression

(You can see it for your self by running [this example](blitz/examples/bayesian_regression_boston.py) on your machine).

We will now see how can Bayesian Deep Learning be used for regression in order to gather confidence interval over our datapoint rather than a pontual continuous value prediction. Gathering a confidence interval for your prediction may be even a more useful information than a low-error estimation. 

I sustain my argumentation on the fact that, with good/high prob a confidence interval, you can make a more reliable decision than with a very proximal estimation on some contexts: if you are trying to get profit from a trading operation, for example, having a good confidence interval may lead you to know if, at least, the value on which the operation wil procees will be lower (or higher) than some determinate X.

Knowing if a value will be, surely (or with good probability) on a determinate interval can help people on sensible decision more than a very proximal estimation that, if lower or higher than some limit value, may cause loss on a transaction. The point is that, sometimes, knowing if there will be profit may be more useful than measuring it.

In order to demonstrate that, we will create a Bayesian Neural Network Regressor for the Boston-house-data toy dataset, trying to create confidence interval (CI) for the houses of which the price we are trying to predict. We will perform some scaling and the CI will be about 75%. It will be interesting to see that about 90% of the CIs predicted are lower than the high limit OR (inclusive) higher than the lower one.

## Importing the necessary modules
Despite from the known modules, we will bring from BLiTZ athe `variational_estimator`decorator, which helps us to handle the BayesianLinear layers on the module keeping it fully integrated with the rest of Torch, and, of course, `BayesianLinear`, which is our layer that features weight uncertanity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

## Loading and scaling data

Nothing new under the sun here, we are importing and standard-scaling the data to help with the training.

```python
X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(np.expand_dims(y, -1))

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=42)


X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
```

# Creating our variational regressor class

We can create our class with inhreiting from nn.Module, as we would do with any Torch network. Our decorator introduces the methods to handle the bayesian features, as calculating the complexity cost of the Bayesian Layers and doing many feedforwards (sampling different weights on each one) in order to sample our loss.

```python
@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 512)
        self.blinear2 = BayesianLinear(512, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)
```

# Defining a confidence interval evaluating function

This function does create a confidence interval for each prediction on the batch on which we are trying to sample the label value. We then can measure the accuracy of our predictions by seeking how much of the prediciton distributions did actually include the correct label for the datapoint.


```python
def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
```

# Creating our regressor and loading data

Notice here that we create our `BayesianRegressor` as we would do with other neural networks.

```python
regressor = BayesianRegressor(13, 1)
optimizer = optim.Adam(regressor.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)
```

## Our main training and evaluating loop

We do a training loop that only differs from a common torch training by having its loss sampled by its sample_elbo method. All the other stuff can be done normally, as our purpose with BLiTZ is to ease your life on iterating on your data with different Bayesian NNs without trouble.

Here is our very simple training loop:

```python
iteration = 0
for epoch in range(100):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()
        
        loss = regressor.sample_elbo(inputs=datapoints,
                           labels=labels,
                           criterion=criterion,
                           sample_nbr=3)
        loss.backward()
        optimizer.step()
        
        iteration += 1
        if iteration%100==0:
            ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
                                                                        X_test,
                                                                        y_test,
                                                                        samples=25,
                                                                        std_multiplier=3)
            
            print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))
            print("Loss: {:.4f}".format(loss))
```

## Bayesian Deep Learning in a Nutshell
A very fast explanation of how is uncertainity introduced in Bayesian Neural Networks and how we model its loss in order to objectively improve the confidence over its prediction and reduce the variance without dropout. 

## First of all, a deterministic NN layer linear transformation

As we know, on deterministic (non bayesian) neural network layers, the trainable parameters correspond directly to the weights used on its linear transformation of the previous one (or the input, if it is the case). It corresponds to the following equation:


![equation](https://latex.codecogs.com/gif.latex?a^{(i&plus;1)}&space;=&space;W^{(i&plus;1)}\cdot&space;z^{(i)}&space;&plus;&space;b^{(i&plus;1)}) 

*(Z correspond to the activated-output of the layer i)*

## The purpose of Bayesian Layers

Bayesian layers seek to introduce uncertainity on its weights by sampling them from a distribution parametrized by trainable variables on each feedforward operation. 

This allows we not just to optimize the performance metrics of the model, but also gather the uncertainity of the network predictions over a specific datapoint (by sampling it much times and measuring the dispersion) and aimingly reduce as much as possible the variance of the network over the prediction, making possible to know how much of incertainity we still have over the label if we try to model it in function of our specific datapoint.

## Weight sampling on Bayesian Layers
To do so, on each feedforward operation we sample the parameters of the linear transformation with the following equations (where **ρ** parametrizes the standard deviation and **μ** parametrizes the mean for the samples linear transformation parameters) :

For the weights:

![equation](https://latex.codecogs.com/gif.latex?W^{(i)}_{(n)}&space;=&space;\mathcal{N}(0,1)&space;*&space;log(1&space;&plus;&space;\rho^{(i)}&space;)&space;&plus;&space;\mu^{(i)})

*Where the sampled W corresponds to the weights used on the linear transformation for the ith layer on the nth sample.*

For the biases:

![equation](https://latex.codecogs.com/gif.latex?b^{(i)}_{(n)}&space;=&space;\mathcal{N}(0,1)&space;*&space;log(1&space;&plus;&space;\rho^{(i)}&space;)&space;&plus;&space;\mu^{(i)})

*Where the sampled b corresponds to the biases used on the linear transformation for the ith layer on the nth sample.*

## It is possible to optimize our trainable weights

Even tough we have a random multiplier for our weights and biases, it is possible to optimize them by, given some differentiable function of the weights sampled and trainable parameters (in our case, the loss), summing the derivative of the function relative to both of them:

1. Let ![equation](https://latex.codecogs.com/gif.latex?\epsilon&space;=&space;\mathcal{N}(0,1))
2. Let ![equation](https://latex.codecogs.com/gif.latex?\theta&space;=&space;(\rho,&space;\mu))
3. Let ![equation](https://latex.codecogs.com/gif.latex?w&space;=&space;\mu&space;&plus;&space;\log({1&space;&plus;&space;e^{\rho}})&space;*&space;\epsilon)
4. Let ![equation](https://latex.codecogs.com/gif.latex?f(w,&space;\theta)) be differentiable relative to its variables

Therefore:

5. ![equation](https://latex.codecogs.com/gif.latex?\Delta_{\mu}&space;=&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;w}&space;&plus;&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;\mu})

and


6. ![equation](https://latex.codecogs.com/gif.latex?\Delta_{\rho}&space;=&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;w}&space;\frac{\epsilon}{1&space;&plus;&space;e^\rho&space;}&space;&plus;&space;\frac{\delta&space;f(w,&space;\theta)}{\delta&space;\rho})

## It is also true that there is complexity cost function differentiable along its variables

It is known that the crossentropy loss (and MSE) are differentiable. Therefore if we prove that there is a complexity-cost function that is differentiable, we can leave it to our framework take the derivatives and compute the gradients on the optimization step.

**The complexity cost is calculated, on the feedforward operation, by each of the Bayesian Layers, (with the layers pre-defined-simpler apriori distribution and its empirical distribution). The sum of the complexity cost of each layer is summed to the loss.**

As proposed in [Weight Uncertainty in Neural Networks paper](https://arxiv.org/abs/1505.05424), we can gather the complexity cost of a distribution by taking the [Kullback-Leibler Divergence](https://jhui.github.io/2017/01/05/Deep-learning-Information-theory/) from it to a much simpler distribution, and by making some approximation, we will can differentiate this function relative to its variables (the distributions):

1. Let ![equation](https://latex.codecogs.com/gif.latex?{P}(w)) be a low-entropy distribution pdf set by hand, which will be assumed as an "a priori" distribution for the weights

2. Let ![equation](https://latex.codecogs.com/gif.latex?{Q}(w&space;|&space;\theta)) be the a posteriori empirical distribution pdf for our sampled weights, given its parameters.




Therefore, for each scalar on the W sampled matrix:




3. ![equation](https://latex.codecogs.com/gif.latex?{D}_{KL}(&space;{Q}(w&space;|&space;\theta)&space;\lVert&space;{P}(w)&space;)&space;=&space;\lim_{n\to\infty}1/n\sum_{i=0}^{n}&space;{Q}(w^{(i)}&space;|&space;\theta)*&space;(\log{{Q}(w^{(i)}&space;|&space;\theta)}&space;-&space;\log{{P}(w^{(i)})}&space;))


By assuming a very large n, we could approximate:

4. ![equation](https://latex.codecogs.com/gif.latex?{D}_{KL}(&space;{Q}(w&space;|&space;\theta)&space;\lVert&space;{P}(w)&space;)&space;=&space;1/n\sum_{i=0}^{n}&space;{Q}(w^{(i)}&space;|&space;\theta)*&space;(\log{{Q}(w^{(i)}&space;|&space;\theta)}&space;-&space;\log{{P}(w^{(i)})}&space;))


and therefore:


5. ![equation](https://latex.codecogs.com/gif.latex?{D}_{KL}(&space;{Q}(w&space;|&space;\theta)&space;\lVert&space;{P}(w)&space;)&space;=&space;\mu_Q&space;*\sum_{i=0}^{n}&space;(\log{{Q}(w^{(i)}&space;|&space;\theta)}&space;-&space;\log{{P}(w^{(i)})}&space;))


As the expected (mean) of the Q distribution ends up by just scaling the values, we can take it out of the equation (as there will be no framework-tracing). Have a complexity cost of the nth sample as:

6. ![equation](https://latex.codecogs.com/gif.latex?{C^{(n)}&space;(w^{(n)},&space;\theta)&space;}&space;=&space;(\log{{Q}(w^{(n)}&space;|&space;\theta)}&space;-&space;\log{{P}(w^{(n)})}&space;))

Which is differentiable relative to all of its parameters. 

## To get the whole cost function at the nth sample:

1. Let a performance (fit to data) function be: ![equation](https://latex.codecogs.com/gif.latex?{P^{(n)}&space;(w^{(n)},&space;\theta)})


Therefore the whole cost function on the nth sample of weights will be:

2. ![equation](https://latex.codecogs.com/gif.latex?{L^{(n)}&space;(w^{(n)},&space;\theta)&space;}&space;=&space;{C^{(n)}&space;(w^{(n)},&space;\theta)&space;}&space;&plus;&space;{P^{(n)}&space;(w^{(n)},&space;\theta)&space;})

We can estimate the true full Cost function by Monte Carlo sampling it (feedforwarding the netwok X times and taking the mean over full loss) and then backpropagate using our estimated value. It works for a low number of experiments per backprop and even for unitary experiments.

## Some notes and wrap up
We came to the end of a Bayesian Deep Learning in a Nutshell tutorial. By knowing what is being done here, you can implement your bnn model as you wish. 

Maybe you can optimize by doing one optimize step per sample, or by using this Monte-Carlo-ish method to gather the loss some times, take its mean and then optimizer. Your move.

FYI: **Our Bayesian Layers and utils help to calculate the complexity cost along the layers on each feedforward operation, so don't mind it to much.**
 
## References:
 * [Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424, 2015.](https://arxiv.org/abs/1505.05424)
 
 
## Citing

If you use `BLiTZ` in your research, you can cite it as follows:

```bibtex
@misc{esposito2020blitzbdl,
    author = {Piero Esposito},
    title = {BLiTZ - Bayesian Layers in Torch Zoo (a Bayesian Deep Learing library for Torch)},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/piEsposito/blitz-bayesian-deep-learning/}},
}
```
 
###### Made by Pi Esposito
