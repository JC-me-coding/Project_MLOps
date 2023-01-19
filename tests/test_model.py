import os.path
import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataloader import load_data
from src.models.model import make_model


root = "data/processed/landscapes"
config = OmegaConf.load("config/train_config.yaml")
backbone = config.model.backbone

# Test the shape of the output from the model


@pytest.mark.skipif(not os.path.exists(root), reason="Data files not found")
def test_model():
    valid_loader = load_data(root, "val", 1, config.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(backbone, pretrained=True).to(device)

    for i, data in enumerate(valid_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        pred = model(images)

        assert pred.shape == (1, 5)

        break


def test_model_github():
    image = torch.rand(3, 224, 224)
    image = image.unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    model = make_model(backbone, pretrained=True).to(device)
    pred = model(image)
    assert pred.shape == (1, 5)
