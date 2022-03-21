import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

from dataloader import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.mnist_model_exp import Classifier, CHead

from math import *
from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal

from PIL import Image, ImageOps
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

seed = 1133
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Load the checkpoint file
state_dict = torch.load(args.load_path, map_location=torch.device('cpu'))

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
classifier = Classifier().to(device)
netC = CHead().to(device)
# Load the trained generator weights.
classifier.load_state_dict(state_dict['classifier'])
netC.load_state_dict(state_dict['netC'])

classifier.train()
netC.train()

feature_extr = nn.Sequential(*list(netC.children())[:-1])
print("Number of features: ", list(feature_extr.parameters())[-1].shape[0])

W = list(netC.fc2.parameters())[0]
shape_W = W.shape
print (W.shape)

_ = extend(netC.fc2)
class_loss_func = extend(nn.CrossEntropyLoss(reduction='sum'))

batch_size = 128

#Get Hessian on training data
dataloader = get_data('MNIST', batch_size, train_test='train')
dataloader_test = get_data('MNIST', 1, train_test='test')

num_samples = 0
for i, (data, true_label) in enumerate(dataloader, 0):
	print ('Batch')
	print (i)

	output_s = classifier(data)
	probs_s = netC(output_s)
	loss_first = class_loss_func(probs_s, true_label)

	with backpack(extensions.KFAC()):
	    loss_first.backward(retain_graph=True)

	if (i == num_samples):
		break

print ('Setting K Fac')
A, B = W.kfac

classifier.eval()
netC.eval()

#Now test
prec0 = 5e-4

A = A.to(device)
B = B.to(device)

eye1 = torch.eye(shape_W[0], device=device)
eye2 = torch.eye(shape_W[1], device=device)

U = torch.inverse(A + sqrt(prec0)*eye1)
V = torch.inverse(B + sqrt(prec0)*eye2)
  
# Read a PIL image
# image = Image.open('8-8.png')

# gray_image = ImageOps.grayscale(image)

# newsize = (28, 28)
# gray_image = gray_image.resize(newsize)

# # Define a transform to convert PIL 
# # image to a Torch tensor
# transform = transforms.Compose([
#     transforms.PILToTensor()
# ])
  
# # transform = transforms.PILToTensor()
# # Convert the PIL image to Torch tensor
# img_tensor = transform(gray_image)

# img_tensor.resize_(1, 1, 28, 28)
# img_tensor = img_tensor.float()

image = torch.load('8-8.pt')
image.resize_(1, 1, 28, 28)
img_tensor = image.float()
img_label = 8

total_correct = 0
total_samples = 0
test_samples = 100
for i, (data, true_label) in enumerate(dataloader_test, 0):

	test_data = img_tensor
	output_s = classifier(test_data)
	output_s = output_s.resize(1, 1024)
	phi = feature_extr(output_s)

	# MAP prediction
	m = phi @ W.T

	# v is the induced covariance. 
	# See Appendix B.1 of https://arxiv.org/abs/2002.10118 for the detail of the derivation.
	v = torch.diag(phi @ V @ phi.T).reshape(-1, 1, 1) * U

	scale_factor = 1000
	v /= scale_factor

	v = torch.eye(10)
	v /= scale_factor
	    
	# The induced distribution over the output (pre-softmax)
	output_dist = MultivariateNormal(m, v)

	# MC-integral
	n_sample = 1000
	probs_split = 0

	for _ in range(n_sample):
	    out_s = output_dist.rsample()
	    probs_split += torch.softmax(out_s, 1)

	probs_split /= n_sample

	print (probs_split)
	print (img_label)
	exit()

	guess = torch.argmax(probs_split, dim=1)

	equal = (true_label == guess)
	total_correct += torch.sum(equal)
	total_samples += 1

	if (i == test_samples):
		break

accuracy = (total_correct/total_samples)
print (accuracy)



