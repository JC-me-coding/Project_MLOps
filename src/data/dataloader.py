from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ToDo: Use more exotic augmentations from timm5
def get_train_transforms(config):
    transform_list = []
    transform_list.append(transforms.RandomResizedCrop(config.input_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    transform_list = []
    # transform_list.append(transforms.Resize(224, 244))# Does not make sense with CenterCrop, right?
    transform_list.append(transforms.CenterCrop(config.input_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_list)


def load_data(root, split, batch_size, data_config):
    if split == "train":
        data = datasets.ImageFolder(f"{root}/Training Data", get_train_transforms(data_config))
    elif split == "val":
        data = datasets.ImageFolder(f"{root}/Validation Data", get_val_transforms(data_config))

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if split == "train" else False,
    )
