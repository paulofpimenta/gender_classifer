# Run a stand alone version of the model and plot its predictions
import os
from model.conv_model import ConvNet
from model.conv_neural_net import ConvolutionalNeuralNet
import torch
from PIL import Image
import urllib

# Instantiating model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net)

# Set working directory to current folder
current_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_folder)

# Load model
model.network.load_state_dict(torch.load("./best_gender_model.pth"))

# Download sample images
# urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11752.png","./data/samples/woman_face.png")
urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11761.png", "./data/samples/man_face.png")
# urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11760.png","./data/samples/man_face2.png")

# Load image with PIL
image = Image.open("./data/samples/man_face.png").convert("RGB")

# Get prediction score
prediction = model.predict(image)

# Plot prediction
model.plot_prediction(prediction, image)
