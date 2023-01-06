import timm
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import re, os, sys, json, random, cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision
import numpy as np



config = OmegaConf.load('config.yaml')

root = "/kaggle/input/landscape-recognition-image-dataset-12k-images/Landscape Classification/Landscape Classification/Training Data/"

def read_split_data(root, plot_image=False):
    filepaths = []
    labels = []
    bad_images = []

    random.seed(0)
    assert os.path.exists(root), 'wdnmd, 你tm路径不对啊!'

    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()
    class_indices = {k: v for v, k in enumerate(classes)}

    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)

    with open('classes_indices.json', 'w') as json_file:
        json_file.write(json_str)

    every_class_num = []
    supported = ['.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for klass in classes:
        classpath = os.path.join(root, klass)
        images = [os.path.join(root, klass, i) for i in os.listdir(classpath) if os.path.splitext(i)[-1] in supported]
        every_class_num.append(len(images))
        flist = sorted(os.listdir(classpath))
        desc = f'{klass:23s}'
        for f in tqdm(flist, ncols=110, desc=desc, unit='file', colour='blue'):
            fpath = os.path.join(classpath, f)
            fl = f.lower()
            index = fl.rfind('.')
            ext = fl[index:]
            if ext in supported:
                try:
                    img = cv2.imread(fpath)
                    filepaths.append(fpath)
                    labels.append(klass)
                except:
                    bad_images.append(fpath)
                    print('defective image file: ', fpath)
            else:
                bad_images.append(fpath)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)

    print(f'{len(df.labels.unique())} kind of images were found in the dataset')
    train_df, test_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])

    train_image_path = train_df['filepaths'].tolist()
    val_image_path = test_df['filepaths'].tolist()

    train_image_label = [class_indices[i] for i in train_df['labels'].tolist()]
    val_image_label = [class_indices[i] for i in test_df['labels'].tolist()]

    sample_df = train_df.sample(n=50, replace=False)
    ht, wt, count = 0, 0, 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h = img.shape[0]
            w = img.shape[1]
            ht += h
            wt += w
            count += 1
        except:
            pass

    have = int(ht / count)
    wave = int(wt / count)
    aspect_ratio = have / wave
    print('{} images were found in the dataset.\n{} for training, {} for validation'.format(
        sum(every_class_num), len(train_image_path), len(val_image_path)
    ))
    print('average image height= ', have, '  average image width= ', wave, ' aspect ratio h/w= ', aspect_ratio)

    if plot_image:
        plt.bar(range(len(classes)), every_class_num, align='center')
        plt.xticks(range(len(classes)), classes)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')
        plt.ylabel('number of images')

        plt.title('class distribution')
        plt.show()

    return train_image_path, train_image_label, val_image_path, val_image_label, class_indices


train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(root, plot_image=True)

class MyDataset(Dataset):
    def __init__(self, image_path, image_labels, transforms=None):
        self.image_path = image_path
        self.image_labels = image_labels
        self.transforms = transforms
        
        
    def __len__(self):
        return len(self.image_path)
    
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        label = self.image_labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label
        
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

batch_size = 32
epochs = 5
weight_decay = 1e-5
lr = 0.001

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(224), 
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

train_dataset = MyDataset(train_image_path, train_image_label, data_transform['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)

valid_dataset = MyDataset(val_image_path, val_image_label, data_transform['valid'])
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=valid_dataset.collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = timm.create_model('resnet34', num_classes=5, pretrained=True)

def train_step(net, optimizer, data_loader, device, epoch):
    net.train()
    loss_function = nn.CrossEntropyLoss()
    train_acc, train_loss, sampleNum = 0, 0, 0
    
    optimizer.zero_grad()
    
    train_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(train_bar):
        images, labels = data
        sampleNum += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        loss.backward()
        optimizer.step()
        train_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, train_loss / (step + 1), train_acc / sampleNum)
    
    return train_loss / (step + 1), train_acc / sampleNum
        
        
@torch.no_grad()
def val_step(net, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()
    net.eval()
    val_acc = 0
    val_loss = 0
    sample_num = 0
    val_bar = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(val_bar):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        val_bar.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, val_loss / (step + 1), val_acc / sample_num)
        
    return val_loss / (step + 1), val_acc / sample_num

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
best_val_acc = 0
for epoch in range(3):
    train_loss, train_accuracy = train_step(net, optimizer, train_loader, device, epoch)
    val_loss, val_accuracy = val_step(net, valid_loader, device, epoch)
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(net.state_dict(), f'./model_{epoch}.pth')

# if __name__ == '__main__':
#     #data
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     batch_size = 32
#     epochs = 5
#     weight_decay = 1e-5
#     lr = 0.001
#     net = timm.create_model('resnet34', num_classes=5, pretrained=True)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

#     best_val_acc = 0

#     for epoch in range(3):
#         train_loss, train_accuracy = train_step(net, optimizer, train_loader, device, epoch)
#         val_loss, val_accuracy = val_step(net, valid_loader, device, epoch)

#         if val_accuracy > best_val_acc:
#             best_val_acc = val_accuracy
#             torch.save(net.state_dict(), f'./model_{epoch}.pth')