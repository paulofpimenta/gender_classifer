


import urllib3
from ConvModel import ConvNet
from classifer import ConvolutionalNeuralNet
import torch
from torchvision import transforms
from mean_std_loader import StatsFromDataSet
from PIL import Image
from urllib.request import urlopen
from PIL import Image
import requests
import numpy as np
from io import StringIO
import urllib.request
from matplotlib import pyplot as plt


conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net)

  #  Instantiating model

model.network.load_state_dict(torch.load('./best_gender_model.pth'))

    
# Get classes and split images into train, test and validation
#train_image_paths,test_image_paths,valid_image_paths,classes = model.select_images_data_sets()
# Create datasets
#train_dataset,test_dataset,valid_dataset = model.create_datasets(train_image_paths,test_image_paths,valid_image_paths,classes)
# Init network weights
#model.network.apply(model.init_weights)
# Create dataloaders
#train_dataloader,test_dataloader,valid_dataloader = model.create_data_loader(train_dataset,test_dataset,valid_dataset)


#std_list = valid_dataloader.dataset.transform.transforms[3].std.tolist()
#mean_list = valid_dataloader.dataset.transform.transforms[3].mean.tolist()

#preprocess = transforms.Compose([
#        transforms.Resize(224),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=mean_list, std=std_list),
#    ])

#input_tensor = preprocess(input_image)

#input, target = valid_dataset[0][0],valid_dataset[0][1]

# Let us look at how the network performs on the whole validation dataset.

#model.evaluate_dataset(valid_dataloader,classes,1)

# To check later

# Download example image
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11752.png","woman_face.png")
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11761.png","man_face.png")
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11760.png","./data/samples/man_face2.png")

# Pre-process image & create a mini-batch as expected by the model
input_image = Image.open("./data/samples/man_face2.png").convert('RGB')

preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 

model.network.eval()
with torch.no_grad():
    output = model.network(input_batch)
    _,prediction = torch.max(output, 1)
    classes=['female','male']
    predicted_class = classes[prediction.item()]
    print(f'Image predicted as {predicted_class.upper()}')
    plt.imshow(input_image)
    plt.title(predicted_class.upper())
    plt.show(block=True)


