import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.transforms import InterpolationMode

# Directory containing the data.
root = 'data/'

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_data(dataset, batch_size, train_test='train', use_3_channel=False):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        if (use_3_channel == False):
            transform = transforms.Compose([
                transforms.Resize(28),
                # transforms.RandomApply(
                #     [
                #         transforms.ColorJitter(
                #             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                #         )
                #     ],
                #     p=0.5,
                # ),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307], std=[0.3081]
                    )])
            # transform = transforms.Compose(
            # [ 
            #     transforms.RandomResizedCrop(
            #         28, scale = (0.9, 1.0), interpolation=InterpolationMode.BILINEAR
            #     ),
            #     transforms.RandomHorizontalFlip(p=0.0),
            #     transforms.RandomApply(
            #         [
            #             transforms.ColorJitter(
            #                 brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            #             )
            #         ],
            #         p=0.2,
            #     ),
            #     transforms.RandomGrayscale(p=0.0),
            #     GaussianBlur(p=0.0),
            #     Solarization(p=0.0),
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.1307], std=[0.3081]
            #     ),
            # ])
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]
                ),]
            )

        # transform = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.Resize(28),
        #     transforms.CenterCrop(28),
        #     transforms.ToTensor()])


        if (train_test == 'train'):
            dataset = dsets.MNIST(root+'mnist/', train=True, 
                                download=True, transform=transform)
        else:
            dataset = dsets.MNIST(root+'mnist/', train=False, 
                                download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train', 
                                download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    elif dataset == 'Cifar10':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                    #mean=[0.49139968, 0.48215841, 0.44653091], 
                    #std=[0.24703223, 0.24348513, 0.26158784]
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ]
            )

        if (train_test == 'train'):
            dataset = dsets.CIFAR10(root=root+'cifar10/', train=True, 
                                    download=True, transform=transform)
        else:
            dataset = dsets.CIFAR10(root=root+'cifar10/', train=False, 
                        download=True, transform=transform)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader