import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

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

@variational_estimator
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(256, 120)
        self.fc2   = BayesianLinear(120, 84)
        self.fc3   = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BayesianCNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

iteration = 0
for epoch in range(100):
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