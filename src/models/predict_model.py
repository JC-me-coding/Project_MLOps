from omegaconf import OmegaConf
from src.model import make_model
import torch
import cv2
import numpy as np


def predict_input(model_weights, image):
  image = cv2.resize(image,[224,224]).astype(np.float32)
  image = torch.from_numpy(image)
  image = image.unsqueeze(0).permute(0,3,1,2)
  config = OmegaConf.load('src/model_config.yaml')
  classes = config.data.classes
  backbone = config.model.backbone
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  image = image.to(device)
  net = make_model(backbone, pretrained=True).to(device)
  net.load_state_dict(torch.load(model_weights))
  net.eval()
  with torch.no_grad():
    prediction = net(image)
    sm = torch.nn.functional.softmax(prediction)
    sm = torch.max(sm,1)
    #print('predicted class:', classes[sm[1].item()], 'with confidence:', sm[0].item())
    return classes[sm[1].item()], sm[0].item()
