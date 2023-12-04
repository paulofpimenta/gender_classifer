"""Defines Neural net architecture"""
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, 7, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm3 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.fc1 = nn.Linear(18816, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10)

        self.apply(self.weights_init)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)
        x = self.norm3(x)

        x = x.view(-1, 18816)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)

        x = F.log_softmax(self.fc3(x), dim=1)

        return x

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1e-2)
