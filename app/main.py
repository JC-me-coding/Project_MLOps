from typing import Optional

import cv2
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile

from src.predict_model import predict_input

app = FastAPI()

@app.post("/predict/")
async def cv_model(data: UploadFile = File(...)):

         with open('image.jpg', 'wb') as image:
            content = await data.read()
            image.write(content)
            image.close()

         # img = cv2.imread("image.jpg")
         img = np.array(Image.open("image.jpg"))

         prediction = predict_input("models/model_best.pth",img)

         return {'prediction': prediction}
         #return {'file_name': data.filename}