#Place to test MNIST models w/ different datasets
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

#Testing grounds for MNIST

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

def test(test_image_path, model, use_custom_dataset, dataset, batch_size, use_custom_model, model_path):
    if (dataset == 'mnist'):
        if (!use_custom_dataset):
            test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size=1, shuffle=True)
    elif (dataset == 'cifar10'):
        if (!use_custom_dataset):
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
    if (model == 'le_net'):
        model = LeNet().to(device)
        if (!use_custom_model):
            model_path = './models/lenet_mnist_model.pth'
    elif (model == 'alex_net'):
        model = AlexNet().to(device)
        if (!use_custom_model)
            #model_path = './models/alexnet_cifar10_model_best.pth.tar'
            model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    else:
        print ('No Valid Model Suggested')
        exit()

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

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        output = model(data)

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
    model = 'le_net'
    dataset = 'mnist'
    train_test = 'test'
    use_custom_dataset = False
    train_image_path = './data/mnist/archive/trainSet/trainSet/*.jpg' #Use only if using Custom Dataset
    test_image_path = './data/mnist/archive/testSet/testSet/*.jpg' #Use only if using Custom Dataset
    use_custom_model = False
    model_path = 0 #Use if using Custom Model
    batch_size = 16

    if (train_test == 'train'):
        test(test_image_path, model, use_custom_dataset, dataset, batch_size, use_custom_model, model_path)
    else:
        test(test_image_path, model, use_custom_dataset, dataset, batch_size, use_custom_model, model_path)