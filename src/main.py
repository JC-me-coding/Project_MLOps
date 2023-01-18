import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataloader import load_data
from src.ml_utils.losses import make_loss_func
from src.models.model import make_model
from src.ml_utils.optimizer import make_optimizer

global config
from timm.optim.sgdp import SGDP


def train_step(net, loss_function, optimizer, data_loader, device, epoch):
    net.train()
    acc_sum, loss_sum, sample_num = 0, 0, 0
    
    optimizer.zero_grad()
    train_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(train_bar):
        iter = epoch * len(data_loader) + step
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss_sum += loss.item()
        acc_sum += (torch.argmax(outputs, dim=1) == labels).sum().item()
        loss.backward()
        optimizer.step()
        loss_step = loss_sum / (step + 1)
        acc_step = acc_sum / sample_num
        train_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, loss_step, acc_step)
        wandb.log({"train/loss": loss_step})
        wandb.log({"train/acc": acc_step})
    
    return loss_sum / (step + 1), acc_sum / sample_num
        
        
@torch.no_grad()
def val_step(net, loss_function, data_loader, device, epoch):
    net.eval()
    acc_sum, loss_sum, sample_num = 0, 0, 0

    val_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(val_bar):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss_sum += loss.item()
        acc_sum += (torch.argmax(outputs, dim=1) == labels).sum().item()

        loss_step = loss_sum / (step + 1)
        acc_step = acc_sum / sample_num

        # Logging
        val_bar.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, loss_step, acc_step)
    wandb.log({"val/loss": loss_step})
    wandb.log({"val/acc": acc_step})
        
    return loss_sum / (step + 1), acc_sum / sample_num

def training(config):
    #ToDo: Put params into code. Quick fix for now, grouping all params
    backbone = config.model.backbone
    batch_size = config.data.batch_size
    optimizer = config.training.optimizer
    epochs = config.training.epochs
    weight_decay = config.hyperparameters.weight_decay
    lr = config.hyperparameters.learning_rate
    num_epochs = config.training.epochs
    loss_fun = config.training.loss_fun
    val_interval = config.training.val_interval

    
    ############# DATA #############
    root = os.getcwd()

    train_loader = load_data(f"{root}/data/processed/landscapes", "train", batch_size, config.data)
    valid_loader = load_data(f"{root}/data/processed/landscapes", "val", batch_size, config.data)
    
    ############# MODEL #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = make_model(backbone, pretrained=True).to(device)
    #net = nn.DataParallel(net)

    ############# LOSS FUNCTION #############
    loss_function = make_loss_func(loss_fun)
    ############# OPTIMIZER #############
  
    if optimizer == "SGDP":
        optimizer = SGDP(net.parameters(),lr=lr)#, net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = make_optimizer(optimizer, net, config)
 
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=wandb_config)
    
    #Magic
    #wandb.watch(net, log_freq=100)
    
    ############# TRAINING #############
    best_val_acc = 0
    for epoch in range(num_epochs):
        wandb.log({"epoch": epoch}, step=epoch*len(train_loader))
        train_loss, train_accuracy = train_step(net, loss_function, optimizer, train_loader, device, epoch)

        if epoch % val_interval == 0:
            val_loss, val_accuracy = val_step(net, loss_function, valid_loader, device, epoch)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(net.state_dict(), f'./models/{backbone}_{config.hyperparameters.learning_rate}.pth')

def train_one_sweep():
    
    with wandb.init():
        wb_config = wandb.config
        for k, v in wb_config.items():
            ks = k.split('.')
            config[ks[0]][ks[1]] = v
        print(config)
        training(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train parser")
    parser.add_argument("-c", "--config", type=str, default="config/train_config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)


    if config.reproducability.apply:
        seed = config.reproducability.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if config.hyperparameters.sweep_config == "":
        training(config)
    else:
        sweep_config = yaml.load(open(config.hyperparameters.sweep_config, "r"), Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=sweep_config, project=f"MLOPsproject-Sweep", entity=config.wandb.entity)
        wandb.agent(sweep_id, function=train_one_sweep, count=15)

 
