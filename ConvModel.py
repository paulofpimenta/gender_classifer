import torch.nn as nn

import torch.nn.functional as F
import torch

class ConvNet(nn.Module):

    #To define an augmentation pipeline, you need to create an instance of the Compose class.
    #As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
    #A call to Compose will return a transform function that will perform image augmentation.
    #(https://albumentations.ai/docs/getting_started/image_augmentation/)

    def __init__(self):
        super().__init__()
        
        # 3 input image channel, 6 output channels, 
	    # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 53 * 53 , 120) # 16 out channels * 53 * 53 from image dimension after two convolutions on 224x224 images
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #print("Input shape = ", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print("Shape after conv1 = ", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("Shape after conv2 = ", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("Shape after flatten = ", x.shape)
        x = F.relu(self.fc1(x)) # relu is an activation function
        #print("Shape after fc1 = ", x.shape)
        x = F.relu(self.fc2(x))
        #print("Shape after fc2 = ", x.shape)
        x = self.fc3(x)
        #print("Final shape = ", x.shape)
        return x