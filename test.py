import torch
import numpy as np
from test_function import ReLTanh

from torchvision import datasets
import torchvision.transforms as transforms

# for reproducibility
SEED = 42

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# split for validation
train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000],
    generator=torch.Generator().manual_seed(SEED))

print(f"There are {len(train_data)} examples in train set.")
print(f"There are {len(valid_data)} examples in valid set.")
print(f"There are {len(test_data)} examples in test set.")

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class FCNN(nn.Module):
    def __init__(self, input_dims=28*28, output_dims=10):
        super(FCNN, self).__init__()

        hidden_dims = 128

        # initialize layers
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.out = nn.Linear(hidden_dims, output_dims)

        # activation functions
        self.nl1 = ReLTanh()
        self.act = nn.Softmax(dim=output_dims)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # [bn, 28, 28] -> [bn, 768]
        x = self.fc1(x)          # [bn, 768] -> [bn, 128]
        x = self.nl1(x)          # [bn, 128] -> [bn, 128]
        x = self.out(x)          # [bn, 128] -> [bn, 10]
        return x

    def classify(self, x):
        x = self(x)
        x = self.act(x)
        return torch.argmax(x, dim=1)

# initialize the NN
model = FCNN()
print(model)

## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# number of epochs to train the model
n_epochs = 30  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)


    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1,
        train_loss
        ))

raise Exception

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
