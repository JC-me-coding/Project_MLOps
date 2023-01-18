import torch
from omegaconf import OmegaConf

from src.data.dataloader import load_data
from src.model import make_model


# Test the shape of the output from the model
def test_model():
    root = "data/processed/landscapes"
    config = OmegaConf.load('src/model_config.yaml')
    backbone = config.model.backbone

    valid_loader = load_data(root, split="val", batch_size=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(backbone, pretrained=True).to(device)

    for i, data in enumerate(valid_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        pred = model(images)

        assert pred.shape == (1, 5)

        break
