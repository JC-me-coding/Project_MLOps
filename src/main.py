import timm
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb


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
        # "optimizer=adam",  # adam, sgd, ... 
    ])

    ############# DATA #############
    # ToDo: Use more exotic augmentations from timm5
    root = "data/raw/landscapes"
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224), 
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_dataset =  datasets.ImageFolder(f"{root}/Training Data", data_transform['train']) 
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_dataset =  datasets.ImageFolder(f"{root}/Validation Data", data_transform['val']) 
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    ############# MODEL #############
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = timm.create_model('resnet34', num_classes=5, pretrained=True)

    ############# LOSS FUNCTION #############
    if config.loss_fun == "l1_loss":
        loss_function = nn.L1Loss()
    else:
        #Fallback to cross entropy
        loss_function = nn.CrossEntropyLoss()

    ############# OPTIMIZER #############
    # ToDo: Use more exotic ones from timm / parametrize
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wandb.init(config=config)
    # Magic
    wandb.watch(net, log_freq=100)
    ############# TRAINING #############
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_step(net, loss_function, optimizer, train_loader, device, epoch)
        val_loss, val_accuracy = val_step(net, loss_function, valid_loader, device, epoch)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), f'./model_{epoch}.pth')