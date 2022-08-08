import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from dataloader import get_data
from utils import *
from config import params

from models.mnist_model_wtsmooth2 import Generator

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        #1 or 3 channels
        #self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor()])

train_set = dsets.MNIST('./train_mnist/', train='train', 
                                download=True, transform=transform)
test_set = dsets.MNIST('./test_mnist/', train=False, 
                                download=True, transform=transform)

batch_size = 128
perc_dataset = .1
subset_samples = int(len(train_set)*perc_dataset)
subset_list = list(range(0, subset_samples, 1))

train_subset = torch.utils.data.Subset(train_set, subset_list)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=True, num_workers=1)


classifier = Encoder()
netG = Generator()

#need to load generator
load_path = './checkpoint/model_load'
gen_state_dict = torch.load(load_path, map_location=device)

load_model = False
load_path = ' '
state_dict = {}
if (load_model):
    load_path = './checkpoint/mnist_load'
    state_dict = torch.load(load_path, map_location=device)

if (load_model):
    classifier.load_state_dict(state_dict['classifier'])

netG.load_state_dict(gen_state_dict['netG'])

#training
train_model = True
use_generator_data = True
num_epochs = 100
eval_epoch = 5
if (train_model):
    for e in range(num_epochs):
        epoch_start_time = time.time()

        for i, (data, true_label) in enumerate(train_loader, 0):
            classifier.train()

            #combine Generator and Subset data
            if (use_generator_data):
                pass


        if (e % eval_epoch == 0):
            total_correct = 0
            total_samples = 0
            for i, (data, true_label) in enumerate(test_loader, 0):
                logits = classifier(data)
                preds = F.softmax(logits, dim=1)

                guess = torch.argmax(probs_c, dim=1)

                equal = (true_label == guess)
                total_correct += torch.sum(equal)
                total_samples += batch_size

            accuracy = (total_correct/total_samples)
            print (accuracy)

#final testing 
print ('Training Done, final test')
total_correct = 0
total_samples = 0
for i, (data, true_label) in enumerate(test_loader, 0):
    logits = classifier(data)
    preds = F.softmax(logits, dim=1)

    guess = torch.argmax(probs_c, dim=1)

    equal = (true_label == guess)
    total_correct += torch.sum(equal)
    total_samples += batch_size

accuracy = (total_correct/total_samples)
print (accuracy)


















