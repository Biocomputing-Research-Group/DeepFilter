import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import time
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import pandas as pd
from datetime import timedelta
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)


class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 7), dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv1.apply(init_weights)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))
        )
        self.conv2.apply(init_weights)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))

        )
        self.conv3.apply(init_weights)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))

        )
        self.conv4.apply(init_weights)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3072, 1024)
        # self.fc2 = nn.Linear(2048,1680)
        self.fc3 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x=self.fc2(x)
        return F.relu(self.fc3(x))
        # return F.softmax(self.fc3(x), dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(11, 512)

    def forward(self, x):
        return F.relu(self.layer1(x))


class mymodel(nn.Module):
    def __init__(self, CNN, Net):
        super(mymodel, self).__init__()
        self.CNN = CNN
        self.Net = Net
        self.fc = nn.Linear(1024, 2)

    def forward(self, x1, x2):
        x1 = self.CNN(x1)
        x2 = self.Net(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)


