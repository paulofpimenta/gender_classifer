import io
from typing import Union
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image

# append the path of the parent directory
import os, sys

from model.ConvModel import ConvNet
from model.classifer import ConvolutionalNeuralNet

#  Instantiating and loac model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net)
model.network.load_state_dict(torch.load('./best_gender_model.pth'))


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
    "http://app1.ouicodedata.com:8000"
]

# Allow these methods to be used
methods = ["GET", "POST"]

# Only these headers are allowed
#headers = ["Content-Type", "Authorization"]


# Declare entry points
@app.get("/api",response_class=JSONResponse)
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins],
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=["*"],
    expose_headers=["*"])