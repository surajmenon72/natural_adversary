#FGSM-Test
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

def test(test_image_path, model_path, batch_size):
	test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize the network
	model = LeNet().to(device)

	# Load the pretrained model
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

	    # Set requires_grad attribute of tensor. Important for Attack
	    data.requires_grad = True

	    #atk = torchattacks.MIFGSM(model, eps=255/255, alpha=255/255, steps=10)
	    atk = mifgsm.MIFGSM(model, eps=128/255, alpha=128/255, steps=100000)
	    atk.set_mode_targeted_least_likely()

	    adv_images = atk(data, target)

	    print (data.shape)
	    print (target.shape)
	    print (adv_images.shape)

	    img_real = data[0].permute(1, 2, 0)
	    img_adv = adv_images[0].permute(1, 2, 0)

	    f, axarr = plt.subplots(2)
	    axarr[0].imshow(img_real.detach().numpy(), cmap='gray')
	    axarr[1].imshow(img_adv.detach().numpy(), cmap='gray')
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
	model_path = './models/lenet_mnist_model.pth'
	batch_size = 16


	test(test_image_path, model_path, batch_size)