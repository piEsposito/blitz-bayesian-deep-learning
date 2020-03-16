import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#imports from our lib
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

#create dataloaders
train_dataset = dsets.MNIST(root="./data",
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="./data",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True
                            )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=32,
                                           shuffle=True)
#lets just create our bnn class
@variational_estimator
class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 512)
        self.blinear2 = BayesianLinear(512, output_dim)
        
    def forward(self, x):
        x_ = x.view(-1, 28 * 28)
        x_ = self.blinear1(x_)
        return self.blinear2(x_)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianNetwork(28*28, 10).to(device)
optimizer = optim.SGD(classifier.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()


iteration = 0
for epoch in range(5):
    for i, (datapoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        loss = classifier.sample_elbo(inputs=datapoints.to(device),
                           labels=labels.to(device),
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
                    outputs = classifier(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                print('Iteration: {} | Accuracy of the network on the 10000 test images: {} %'.format(str(iteration) ,str(100 * correct / total)))
                