from omegaconf import OmegaConf
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from src.model import make_model
from src.data import make_data
from src.optimizer import make_optimizer
from src.losses import make_loss_func


def train_step(net, loss_function, optimizer, data_loader, device, epoch):
    net.train()
    acc_sum, loss_sum, sample_num = 0, 0, 0
    
    optimizer.zero_grad()
    
    train_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(train_bar):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
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

    
    return train_loss / (step + 1), acc_sum / sample_num
        
        
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


if __name__ == '__main__':

    ############# CONFIG #############
    # ToDo: Extract config
    config = OmegaConf.from_dotlist([
        "batch_size=32",
        "epochs=5",
        "weight_decay=1e-5",
        "lr=0.001",
        "num_epochs=3",
        "loss_fun=cross_entropy",  # l1_loss, cross_entropy, ...
        "val_interval=5"
        "optimizer=adam",  # adam, sgd, ... 
    ])

    ############# DATA #############
    root = "data/raw/landscapes"
    
    train_dataset = make_data(root, split="train")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_dataset = make_data(root, split="val")
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    ############# MODEL #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = make_model('resnet34', pretrained=True)

    ############# LOSS FUNCTION #############
    loss_function = make_loss_func(config.loss_fun)

    ############# OPTIMIZER #############
    optimizer = make_optimizer(config.optimizer, net, config)

    wandb.init(config=config)
    # Magic
    wandb.watch(net, log_freq=100)
    
    ############# TRAINING #############
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_step(net, loss_function, optimizer, train_loader, device, epoch)

        if epoch % config.val_interval == 0:
            val_loss, val_accuracy = val_step(net, loss_function, valid_loader, device, epoch)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(net.state_dict(), f'./model_{epoch}.pth')