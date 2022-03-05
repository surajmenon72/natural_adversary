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

seed = 1123
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

classifier.eval()
netC.eval()

dataloader = get_data('MNIST', 1)

total_correct = 0
total_samples = 0
for i, (data, true_label) in enumerate(dataloader, 0):
	real_data = data.to(device)
	output_c = classifier(real_data)
	probs_c = netC(output_c)
	probs_c = torch.squeeze(probs_c)

	guess = torch.argmax(probs_c)
	truth = true_label[0]

	if (guess == truth):
		total_correct += 1

	total_samples += 1

accuracy = (total_correct/total_samples)
print (accuracy)



