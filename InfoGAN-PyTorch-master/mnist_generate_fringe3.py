import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.mnist_model_exp import Generator

seed = 1125
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
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

netG.eval()

z = torch.randn(2, 62, 1, 1, device=device)
c1 = torch.zeros((2, 10, 1, 1))
c2 = torch.zeros((2, 10, 1, 1))

num_ind = 2
cont_ind = 8
cont_ind_val = 0
c1[0, num_ind, 0, 0] = 1
c2[0, cont_ind, 0, 0] = cont_ind_val
# c1[1, 2, 0, 0] = 1
# c2[1, cont_ind, 0, 0] = 0

# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)


# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()

img_to_show = generated_img1[0].permute(1, 2, 0)
plt.figure(1)
plt.axis("off")
plt.imshow(img_to_show)
plt.show()
