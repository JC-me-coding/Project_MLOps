import torch
from timm.data.auto_augment import rand_augment_transform
import torchvision.transforms as transforms
import torchvision.datasets as datasets


if torch.cuda.is_available():
    print(">>> Torch correctly installed, CUDA is available!")
else:
    print(">>> Torch correctly installed, no CUDA, using CPU!")
    #transform_list.append(rand_augment_transform(config_str = 'rand-m9-n3-mstd0.5', hparams = {}))

randomcrop_totensor_transform = transforms.Compose([(rand_augment_transform(config_str = 'rand-m9-n3-mstd0.5', hparams = {})),
                                                    transforms.CenterCrop(16),
                                                    transforms.ToTensor()])
dataset = datasets.FakeData(size=100, image_size=(32, 32), num_classes=10, transform=randomcrop_totensor_transform)

print(">>> Torchvision and Timm correctly installed, you are good to go!")
