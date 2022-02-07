#Grad-Adv-Test
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
#import torchattacks
import glob
import sys
import os

sys.path.append(os.path.abspath("/Users/surajmenon/Desktop/natural_adversary/adversarial-attacks-pytorch-master"))
from torchattacks_local import attack
from torchattacks_local.attacks import mifgsm
from torchattacks_local.attacks import fib_attack

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

#Generating Adversaries from MNIST and Others using TorchAttacks

#Model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def semi_forward(self, x):
        y = F.max_pool2d(self.conv1(x), 2)
        ya = F.relu(y)
        z = F.max_pool2d(self.conv2_drop(self.conv2(ya)), 2)
        za = F.relu(z)

        # x = za.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1), y, ya, z, za

        q = za.view(-1, 320)
        q = F.relu(self.fc1(q))
        q = F.dropout(q, training=self.training)
        q = self.fc2(q)
        return F.log_softmax(q, dim=1), y, ya, z, za


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test(test_image_path, model, dataset, batch_size):
    if (dataset == 'mnist'):
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    elif (dataset == 'cifar10'):
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    else:
        print ('No Valid Dataset Suggested')
        exit()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the network
    model_path = 0
    if (model == 'le_net'):
        model = LeNet().to(device)
        model_path = './models/lenet_mnist_model.pth'
    elif (model == 'alex_net'):
        # model = AlexNet().to(device)
        # model_path = './models/alexnet_cifar10_model_best.pth.tar'
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    else:
        print ('No Valid Model Suggested')
        exit(())

    # Load the pretrained model
    if (model_path != 0):
        model.load_state_dict(torch.load(model_path, map_location=device))

    print ('Model Loaded')

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    # Accuracy counter
    correct = 0
    adv_examples = []

    print ('Starting')

    #atk = torchattacks.MIFGSM(model, eps=255/255, alpha=255/255, steps=10)
    #atk = mifgsm.MIFGSM(model, eps=255/255, alpha=255/255, steps=1000)
    atk = fib_attack.FIBA(model, train_loader, eps=255/255, alpha=128/255, steps=10)
    atk.set_mode_targeted_least_likely()
    atk.set_fi(10)

    atk_exp = fib_attack.FIBA(model, train_loader, eps=255/255, alpha=128/255, steps=10, mode='Step-High')
    atk_exp.set_mode_targeted_least_likely()
    atk_exp.set_fi(10)

    atk_exp_2 = fib_attack.FIBA(model, train_loader, eps=255/255, alpha=128/255, steps=10, mode='Step-Low')
    atk_exp_2.set_mode_targeted_least_likely()
    atk_exp_2.set_fi(10)


    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        adv_images = atk(data, target)
        adv_images_exp = atk_exp(data, target)
        adv_images_exp_2 = atk_exp_2(data, target)

        img_real = data[0].permute(1, 2, 0)
        img_adv = adv_images[0].permute(1, 2, 0)
        img_adv_exp = adv_images_exp[0].permute(1, 2, 0)
        img_adv_exp_2 = adv_images_exp_2[0].permute(1, 2, 0)

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(img_real.detach().numpy(), cmap='gray')
        axarr[0, 1].imshow(img_adv.detach().numpy(), cmap='gray')
        axarr[1, 0].imshow(img_adv_exp.detach().numpy(), cmap='gray')
        axarr[1, 1].imshow(img_adv_exp_2.detach().numpy(), cmap='gray')
        plt.show()
        exit()

        output = model(adv_images)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if (final_pred.item() == target.item()):
            print ('Correct')
            correct += 1
        else:
            print ('Incorrect')

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
    test_image_path = './data/mnist/archive/testSet/testSet/*.jpg' #for now dl using torchvision
    model = 'le_net'
    dataset = 'mnist'
    batch_size = 16


    test(test_image_path, model, dataset, batch_size)