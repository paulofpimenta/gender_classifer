import io
from typing import Union
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
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
    "http://localhost:3000",
]

# Allow these methods to be used
methods = ["GET", "POST"]

# Only these headers are allowed
#headers = ["Content-Type", "Authorization"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=["*"],
)

# Declare entry points
@app.get("/",response_class=JSONResponse)
def read_root():
    response = {"Hello": "World"}
    return response


@app.post("/image")
async def upload_file(file: Union[UploadFile, None] = None):
    if not file:
        print("File type :", type(file))
        return {"message": "No upload file sent"}
    else:
        request_object_content = await file.read()

        input_image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
             
        prediction = model.predict(input_image)
        return JSONResponse(prediction)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}