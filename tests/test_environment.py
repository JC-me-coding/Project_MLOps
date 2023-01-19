import sys


REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version
            )
        )
    else:
        print(">>> Development environment passes all tests!")
    try:
        import torch
        from timm.data.auto_augment import rand_augment_transform
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

        if torch.cuda.is_available():
            print(">>> Torch correctly installed, CUDA is available!")
        elif torch.cuda.is_available() == False:
            print(">>> Torch correctly installed, no CUDA, using CPU!")
        else:
            raise TypeError(
                "Pytorch not configure correctly, please check your installation."
            )
        try:
            randomcrop_totensor_transform = transforms.Compose(
                [
                    (
                        rand_augment_transform(
                            config_str="rand-m9-n3-mstd0.5", hparams={}
                        )
                    ),
                    transforms.CenterCrop(16),
                    transforms.ToTensor(),
                ]
            )
            dataset = datasets.FakeData(
                size=100,
                image_size=(32, 32),
                num_classes=10,
                transform=randomcrop_totensor_transform,
            )
            print(">>> Torchvision and Timm correctly installed, you are good to go!")
        except:
            raise TypeError(
                "Torchvision or timm not configured correctly. Please check your installation."
            )

    except ImportError as impErr:
        print("[Error]: Failed to import {}.".format(impErr.args[0]))
        sys.exit(1)


if __name__ == "__main__":
    main()
