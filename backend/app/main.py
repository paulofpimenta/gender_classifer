import io
from typing import Union
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import os

from model.conv_model import ConvNet
from model.conv_neural_net import ConvolutionalNeuralNet

#  Instantiating and load model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net)

dir_path = os.path.dirname(os.path.realpath(__file__))
model.network.load_state_dict(
    torch.load(dir_path + "/best_gender_model.pth", map_location=torch.device("cpu"))
)


# Start Api and add CORS exceptions
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://app1.ouicodedata.com" "https://app1.ouicodedata.com",
]

# Allow these methods to be used
methods = ["GET", "POST"]

# Only these headers are allowed
# headers = ["Content-Type", "Authorization"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=["*"],
    expose_headers=["*"],
)


# Declare entry points
@app.get("/api", response_class=JSONResponse)
def read_root():
    response = {"Hello": "World"}
    return response


@app.post("/api/image")
async def upload_file(file: Union[UploadFile, None] = None):
    if not file:
        print("File type :", type(file))
        return {"message": "No upload file sent"}
    else:
        request_object_content = await file.read()
        input_image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        prediction = model.predict(input_image)
        return JSONResponse(prediction)
