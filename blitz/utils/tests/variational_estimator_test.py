import unittest
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianConv2d, BayesianLinear, BayesianLSTM, BayesianEmbedding, GaussianVariational
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
                               sample_nbr=5,
                               complexity_cost_weight=1)


        elbo = net.sample_elbo(inputs=batch[0],
                               labels=batch[1],
                               criterion=torch.nn.CrossEntropyLoss(),
                               sample_nbr=5,
                               complexity_cost_weight=0)

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

        net.freeze_()
        self.assertEqual((net(batch[0])==net(batch[0])).all(), torch.tensor(True))

        net.unfreeze_()
        self.assertEqual((net(batch[0])!=net(batch[0])).any(), torch.tensor(True))
        pass

    def test_moped(self):

        @variational_estimator
        class BayesianMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.blinear1 = BayesianLinear(10, 512)
                self.bconv = BayesianConv2d(3, 3, kernel_size=(3, 3), padding=1, bias=True)
                self.blstm = BayesianLSTM(10, 2)
            def forward(self, x):
                return x
        model = BayesianMLP()
        model.MOPED_()

    def test_mfvi(self):

        @variational_estimator
        class BayesianMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.nn = nn.Sequential(BayesianLinear(10, 7), 
                                        BayesianLinear(7, 5))
            def forward(self, x):
                return self.nn(x)
        
        net = BayesianMLP()
        t = torch.ones(3, 10)
        out_ = net(t)

        mean_, std_ = net.mfvi_forward(t, sample_nbr=5)
        self.assertEqual(out_.shape, mean_.shape)
        self.assertEqual(out_.shape, std_.shape)

        self.assertEqual((out_!=mean_).any(), torch.tensor(True))
        self.assertEqual((std_!=0).any(), torch.tensor(True))

        #we also check if, for the frozen model, the std is 0 and the mean is equal to any output
        net.freeze_()
        out__ = net(t)

        mean__, std__ = net.mfvi_forward(t, sample_nbr=5)

        self.assertEqual(out__.shape, mean__.shape)
        self.assertEqual(out__.shape, std__.shape)

        self.assertEqual((std__==0).all(), torch.tensor(True))


if __name__ == "__main__":
    unittest.main()