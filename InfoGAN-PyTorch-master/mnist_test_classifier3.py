import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

from dataloader import get_data


# parser = argparse.ArgumentParser()
# parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
# args = parser.parse_args()

#from models.mnist_model_exp import Classifier, CHead
from models.mnist_model_wtsmooth2 import Encoder, ResnetEncoder, CHead


seed = 1125
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Load the checkpoint file
path = './checkpoint/model_c_load'
state_dict = torch.load(path, map_location=torch.device('cpu'))

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
#classifier = Encoder().to(device)
classifier = ResnetEncoder().to(device)
netC = CHead().to(device)
# Load the trained generator weights.
classifier.load_state_dict(state_dict['classifier'])
netC.load_state_dict(state_dict['netC'])

classifier.eval()
netC.eval()

image = torch.load('8-8.pt')
image = torch.cat([image, image, image], dim=0) 
image.resize_(1, 3, 28, 28)
img_tensor = image.float()

batch_size = 128

dataloader = get_data('MNIST', batch_size, train_test='test')

total_correct = 0
total_samples = 0
for i, (data, true_label) in enumerate(dataloader, 0):
	print ('Batch')
	print (i)
	real_data = data.to(device)
	real_data = img_tensor
	output_c = classifier(real_data)
	probs_c = netC(output_c)
	probs_c = F.softmax(probs_c, dim=1)

	print (probs_c[0])
	exit()

	guess = torch.argmax(probs_c, dim=1)
	 
	equal = (true_label == guess)
	total_correct += torch.sum(equal)
	total_samples += batch_size

accuracy = (total_correct/total_samples)
print (accuracy)



