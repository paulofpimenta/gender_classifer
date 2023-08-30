import io
from typing import Union
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add root folder to system path
root_folder = os.path.abspath(os.path.join(__file__ ,"../../../model"))
sys.path.append(root_folder)

from model import ConvModel,classifer

#  Instantiating and loac model
conv_net = ConvModel.ConvNet()
model = classifer.ConvolutionalNeuralNet(conv_net)
model.network.load_state_dict(torch.load('./best_gender_model.pth'))


# Start Api and add CORS exceptions
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
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
        return {"message": "No upload file sent"}
    else:
        request_object_content = await file.read()
        
        input_image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
             
        prediction = model.predict(input_image)

        return {"filename": file.filename, "prediction" : prediction}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}