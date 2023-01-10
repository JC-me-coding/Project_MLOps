from torchvision import datasets, transforms

# ToDo: Use more exotic augmentations from timm5
def get_train_transforms():
    transform_list = []
    transform_list.append(transforms.RandomResizedCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def get_val_transforms():
    transform_list = []
    transform_list.append(transforms.Resize(224, 244))
    transform_list.append(transforms.CenterCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def make_data(root, split="train"):
    if split == "train":
        datasets.ImageFolder(f"{root}/Training Data", get_train_transforms()) 
    elif split == "val":
        datasets.ImageFolder(f"{root}/Validation Data", get_val_transforms()) 
