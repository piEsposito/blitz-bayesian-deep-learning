import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn

train_dataset = dsets.MNIST(root="./data",
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="./data",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True
                            )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=64,
                                           shuffle=True)


class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.conv_net = nn.Sequential(BayesianConv2d(in_channels=1,
                                                out_channels=32,
                                                kernel_size=(3,3)),
                                     nn.ReLU(),
                                     BayesianConv2d(in_channels=32,
                                                out_channels=64,
                                                kernel_size=(3,3)),
                                     nn.ReLU())
        
        self.fc1 = BayesianLinear(36864, 128)
        self.fc2 = BayesianLinear(128, 10)
        
    def forward(self, x):
        x_ = self.conv_net(x)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.fc1(x_)
        return self.fc2(x_)

classifier = BayesianCNN()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

iteration = 0
for epoch in range(100):
    for i, (datapoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = classifier.sample_elbo(inputs=datapoints,
                           labels=labels,
                           criterion=criterion,
                           sample_nbr=3)
        loss.backward()
        optimizer.step()
        
        iteration += 1
        if iteration%1000==0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = classifier(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Iteration: {} | Accuracy of the network on the 10000 test images: {} %'.format(str(iteration) ,str(100 * correct / total)))