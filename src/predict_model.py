import cv2
import PIL
import numpy as np
from src.models.model import make_model
import torch
from omegaconf import OmegaConf
from src.data.dataloader import get_val_transforms


def predict_input(model_weights, image):
    config = OmegaConf.load("config/train_config.yaml")
    val_transforms = get_val_transforms(config.data)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    classes = config.data.classes
    backbone = config.model.backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    net = make_model(backbone, pretrained=True).to(device)
    net.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))
    net.eval()
    with torch.no_grad():
        prediction = net(image)
        sm = torch.nn.functional.softmax(prediction)
        sm = torch.max(sm, 1)
        # print('predicted class:', classes[sm[1].item()], 'with confidence:', sm[0].item())
        return classes[sm[1].item()], sm[0].item()
