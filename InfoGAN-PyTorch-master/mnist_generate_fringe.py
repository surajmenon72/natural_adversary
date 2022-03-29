import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.mnist_model_smooth import Generator

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
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

netG.eval()

start = 1
stop = 3
#c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.linspace(start, stop, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(100, 1, 1, 1, device=device)

# Continuous latent code.
# c2 = torch.cat((zeros, zeros), dim=1)
# c2 = torch.cat((c2, zeros), dim=1)
# c2 = torch.cat((c2, ), dim=1)

# for i in range(3, 10):
# 	c2 = torch.cat((c2, zeros), dim=1)

c_index = 0
c_index2 = 2
reduction_factor = 1
#c2 = torch.zeros((10, 100, 1, 1), device=device)
c2 = torch.zeros((1, 100, 1, 1), device=device)
#primary
c2[c_index] = c[:, :, 0]*reduction_factor
#secondary
#c2[c_index2] = (c[:, :, 0])/reduction_factor
c2 = c2.permute(1, 0, 2, 3)

idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch.arange(0, 100), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1)

z = torch.randn(100, 62, 1, 1, device=device)

# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)

# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()

print (generated_img1.shape)
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
save_str = str(c_index) + '.png'
plt.savefig(save_str)
plt.show()

# fig_save = np.transpose(generated_img1[65], (1, 2, 0))

# plt.imshow(fig_save)
# plt.show()

# img_to_save = generated_img1[65]
# print (img_to_save.shape)
# torch.save(img_to_save, '7-8.pt')
