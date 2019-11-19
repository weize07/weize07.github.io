# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim as optim
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class FashionCNN(nn.Module):

    # input size of out data is 1 * 28 * 28
    # for conv: Wout = (Win-F+2P)/S + 1
    # for pool: Wout = (Win-F)/S+1
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1) #Wout = 28
        self.add_module('Conv1', self.conv1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.add_module('ConvBN1', self.conv1_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #Wout=14
        self.add_module('Pool1', self.pool1)
        self.relu1 = nn.ReLU()
        self.add_module('Relu1', self.relu1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #Wout=14
        self.add_module('Conv2', self.conv2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.add_module('ConvBN2', self.conv2_bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #Wout=7
        self.add_module('Pool2', self.pool2)
        self.relu2 = nn.ReLU()
        self.add_module('Relu2', self.relu2)
        self.fc1 = nn.Linear(7 * 7 * 128, 256) #input: width and height = Wout of pool2, depth = 18
        self.add_module('FC1', self.fc1)
        self.conv3_bn = nn.BatchNorm1d(256)
        self.add_module('ConvBN3', self.conv3_bn)
        self.relu3 = nn.ReLU()
        self.add_module('Relu3', self.relu3)
        self.fc2 = nn.Linear(256, 10) # output: number of classes
        self.add_module('FC2', self.fc2)
        self.soft = nn.Softmax()
        self.add_module('Softmax', self.soft)

    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu2(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 128)

        x = self.relu3(self.conv3_bn(self.fc1(x)))

        x = self.soft(self.fc2(x))
        return x



if __name__ == '__main__':
    root = './data'
    learning_rate = 0.001
    n_epochs = 30
    batch_size = 64
    train_ds = datasets.FashionMNIST(root, train=True, transform = transforms.ToTensor(), target_transform=None, download=False)
    test_ds = datasets.FashionMNIST(root, train=False, transform = transforms.ToTensor(), target_transform=None, download=False)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    cnn = FashionCNN()
    # layers = []
    # layers.append(nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1))
    # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
    # layers.append(nn.ReLU())
    # layers.append(nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1))
    # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
    # layers.append(nn.ReLU())
    # layers.append(nn.Linear(7 * 7 * 18, 64))
    # layers.append(nn.ReLU())
    # layers.append(nn.Linear(nn.Linear(64, 10)))
    # layers.append(nn.Softmax())

    # cnn = nn.Sequential(*layers)

    cnn.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    n_batches = len(train_loader)

    training_start_time = time.time()

    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()

            #Forward pass, backward pass, optimize
            outputs = cnn(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        total_correct = 0
        total = 0
        for inputs, labels in val_loader:
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()

            #Forward pass
            val_outputs = cnn(inputs)
            winners = val_outputs.argmax(dim=1)
            corrects = (winners == labels)
            # print(corrects.sum().item())

            total_correct += corrects.sum().item()
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data
            total += list(corrects.size())[0]

        print(total_correct, len(val_loader))    
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("Validation accuracy = {:.3f}".format(total_correct / total))



