import unittest
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

class TestVariationalInference(unittest.TestCase):

    def test_kl_divergence(self):
        #create model
        #do two inferences over same datapoint, check if different

        to_feed = torch.ones((1, 10))

        @variational_estimator
        class VariationalEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.blinear = BayesianLinear(10, 10)

            def forward(self, x):
                return self.blinear(x)

        model = VariationalEstimator()
        predicted = model(to_feed)

        complexity_cost = model.nn_kl_divergence()
        kl_complexity_cost = kl_divergence_from_nn(model)

        self.assertEqual((complexity_cost == kl_complexity_cost).all(), torch.tensor(True))

    def test_elbo_sampler(self):
        dataset = dsets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True
                                    )

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=16,
                                                   shuffle=True)

        batch = next(iter(dataloader))

        @variational_estimator
        class BayesianMLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                #self.linear = nn.Linear(input_dim, output_dim)
                self.blinear1 = BayesianLinear(input_dim, 512)
                self.blinear2 = BayesianLinear(512, output_dim)
                
            def forward(self, x):
                x_ = x.view(-1, 28 * 28)
                x_ = self.blinear1(x_)
                return self.blinear2(x_)

        net = BayesianMLP(28*28, 10)
        elbo = net.sample_elbo(inputs=batch[0],
                               labels=batch[1],
                               criterion=torch.nn.CrossEntropyLoss(),
                               sample_nbr=5)

        self.assertEqual((elbo==elbo).all(), torch.tensor(True))
        
        pass

    def test_freeze_estimator(self):
        #create model, freeze it
        #infer two times on same datapoint, check if all equal
        dataset = dsets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True
                                    )

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=16,
                                                   shuffle=True)

        batch = next(iter(dataloader))

        @variational_estimator
        class BayesianMLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                #self.linear = nn.Linear(input_dim, output_dim)
                self.blinear1 = BayesianLinear(input_dim, 512)
                self.blinear2 = BayesianLinear(512, output_dim)
                
            def forward(self, x):
                x_ = x.view(-1, 28 * 28)
                x_ = self.blinear1(x_)
                return self.blinear2(x_)

        net = BayesianMLP(28*28, 10)
        self.assertEqual((net(batch[0])!=net(batch[0])).any(), torch.tensor(True))

        net.freeze()
        self.assertEqual((net(batch[0])==net(batch[0])).all(), torch.tensor(True))

        net.unfreeze()
        self.assertEqual((net(batch[0])!=net(batch[0])).any(), torch.tensor(True))
        pass

if __name__ == "__main__":
    unittest.main()