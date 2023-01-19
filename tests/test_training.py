import os.path
import pytest
import torch
import wandb
from omegaconf import OmegaConf

from src.data.dataloader import load_data
from src.ml_utils.losses import make_loss_func
from src.main import train_step, val_step
from src.models.model import make_model
from src.ml_utils.optimizer import make_optimizer

root = "data/processed/landscapes"
config = OmegaConf.load("config/train_config.yaml")
batch_size = config.data.batch_size
backbone = config.model.backbone


@pytest.mark.skipif(not os.path.exists(root), reason="Data files not found")
def test_training():
    optimizer = config.training.optimizer
    loss_function = config.training.loss_fun
    epoch = 1
    train_loader = load_data(root, "train", batch_size, config.data)
    valid_loader = load_data(root, "val", 1, config.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(backbone, pretrained=True).to(device)

    optimizer = make_optimizer(optimizer, model, config)
    loss_function = make_loss_func(loss_function)

    wandb.init(mode="disabled")

    train_loss, train_accuracy = train_step(
        model, loss_function, optimizer, train_loader, device, epoch
    )

    assert train_loss != 0.0

    val_loss, val_accuracy = val_step(model, loss_function, valid_loader, device, epoch)

    assert val_loss == val_loss
