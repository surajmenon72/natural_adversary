import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Directory containing the data.
root = 'data/'

def get_data(dataset, batch_size, train_test='train', use_3_channel=False):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        if (use_3_channel == False):
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor()])
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
            dataset = dsets.MNIST(root+'mnist/', train='train', 
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

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader