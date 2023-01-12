# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    torch.manual_seed(0)
    batch_size = 64

    # training transforms
    train_transform = transforms.Compose(
        [
            # transforms.Resize((x,y)), #TODO??
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.RandomVerticalFlip(p=0.1),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomRotation(degrees=(10, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # validation transforms
    valid_transform = transforms.Compose(
        [
            # transforms.Resize((x,y)), #TODO??
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # training dataset
    train_dataset = datasets.ImageFolder(
        root=input_filepath + "/Training Data", transform=train_transform
    )
    # validation dataset
    valid_dataset = datasets.ImageFolder(
        root=input_filepath + "/Validation Data", transform=valid_transform
    )
    # training data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # validation data loaders
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # save dataloaders
    torch.save(train_loader, output_filepath + "/train_dl.pt")
    torch.save(valid_loader, output_filepath + "/validation_dl.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found,
    # then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
