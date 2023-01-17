from fastapi import FastAPI, UploadFile, File
from typing import Optional
import cv2
import numpy as np
from src.models.predict_model import predict_input

app = FastAPI()

@app.post("/predict/")
async def cv_model(data: UploadFile = File(...)):

         with open('image.jpg', 'wb') as image:
            content = await data.read()
            image.write(content)
            image.close()

         img = cv2.imread("image.jpg")

         prediction = predict_input(0,img)

         return {'prediction': prediction}
         #return {'file_name': data.filename}