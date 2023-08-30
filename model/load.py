from ConvModel import ConvNet
from classifer import ConvolutionalNeuralNet
import torch
from PIL import Image
import os

# Instantiating model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net)

# Set working directory
current_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_folder)

model.network.load_state_dict(torch.load('./best_gender_model.pth'))

# Download example image
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11752.png","woman_face.png")
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11761.png","man_face.png")
#urllib.request.urlretrieve("https://pngimg.com/uploads/face/face_PNG11760.png","./data/samples/man_face2.png")

# Load image with PIL
image = Image.open("./data/samples/woman_face.png").convert('RGB')

# Get prediction score
prediction = model.predict(image)

# Plot prediction
model.plot_prediction(prediction,image)
