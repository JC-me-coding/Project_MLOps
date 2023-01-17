from omegaconf import OmegaConf
import torch
from src.model import make_model
from src.data.dataloader import load_data
from src.main import train_step
from src.main import val_step
from src.optimizer import make_optimizer
from src.losses import make_loss_func
import wandb


def test_training():
    root = "data/processed/landscapes"
    config = OmegaConf.load('src/model_config.yaml')
    batch_size = config.data.batch_size
    backbone = config.model.backbone
    loss_function = config.training.loss_fun
    optimizer = config.training.optimizer
    epoch = 1

    valid_loader = load_data(root, split="val", batch_size=1)
    train_loader = load_data(root, split="train", batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(backbone, pretrained=True).to(device)

    optimizer = make_optimizer(optimizer, model, config)
    loss_function = make_loss_func(loss_function)

    wandb.init(mode='disabled')

    train_loss, train_accuracy = train_step(model, loss_function, optimizer, train_loader, device, epoch)

    assert train_loss != 0.0

    val_loss, val_accuracy = val_step(model, loss_function, valid_loader, device, epoch)

    assert val_loss == val_loss
