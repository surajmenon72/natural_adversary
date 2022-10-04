import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from dataloader import get_data
from utils import *
from config import params

from models.mnist_model_wtsmooth2 import Encoder, CHead, Generator
from copy import deepcopy

# Set random seed for reproducibility.
seed = 1130
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device('cpu')
print(device, " will be used.\n")

train_eval = 'train'

load_model = True

use_base_resnet = 'base'
use_thanos_vicreg = 'thanos'
load_encoder = True

load_path = ' '
state_dict = {}
if (load_model):
    load_path = './checkpoint/model_load'
    state_dict = torch.load(load_path, map_location=device)

use_3_channel = False
if (use_base_resnet == 'resnet'):
    use_3_channel = True

dataloader = get_data('FashionMNIST', params['batch_size'], train_test='train_index', use_3_channel=use_3_channel, do_shuffle=False)
dataloader_eval = get_data('FashionMNIST', params['batch_size'], train_test='eval_index', use_3_channel=use_3_channel, do_shuffle=False)


# Initialise the network, KNN network
classifier = Encoder().to(device)
classifier.apply(weights_init)
print (classifier)

netC = CHead().to(device)
netC.apply(weights_init)
print (netC)

netG = Generator().to(device)
netG.apply(weights_init)
print (netG)

split_measure = nn.KLDivLoss()

if (load_model):
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
    if (train_eval == 'train'):
        netG.load_state_dict(state_dict['netG'])
    print ('Model successfully loaded')


print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0
num_batches = len(dataloader)

if (train_eval == 'train'):
    b_size = params['batch_size']
    gen_images = torch.zeros((b_size, 1, 28, 28))
    gen_labels = torch.zeros((b_size, 10))

    full_images = torch.zeros((1, 1, 28, 28))
    full_labels = torch.zeros((1, 10))

    num_passes = 1
    for p in range(num_passes):
        epoch_start_time = time.time()

        classifier.eval()
        netC.eval()
        netG.eval()

        for i, (data, true_label, idx) in enumerate(dataloader_train, 0):
            real_data = data.to(device)
            true_label_g = true_label.to(device)

            embedding = classifier(real_data)

            for j in range(b_size):
                if (j < (b_size-1)):
                    i0 = j
                    i1 = j+1
                else:
                    i0 = j
                    i1 = 0

                diff = embedding[i1] - embedding[i0]
                diff /= 2
                noise = embedding[i0] + diff

                gen_images[i0] = netG(noise).detach().cpu()
                l0 = true_label_g[i0]
                l1 = true_label_g[i1]
                gen_labels[i0, l0] = 0.5
                gen_labels[i0, l1] = 0.5

            full_images = torch.cat((full_images, gen_images), dim=0)
            full_labels = torch.cat((full_labels, gen_labels), dim=0)

    torch.save({
        'images': full_images[1:],
        'labels' : full_labels[1:]
        } 'checkpoint/gen_fmnist_%d' % seed)
else:
    load_path = './checkpoint/gen_fmnist_1130'
    gen_dset = torch.load(load_path, map_location=device)

    images = gen_dset['images'].to(device)
    labels = gen_dset['labels'].to(device)

    total_kl = 0
    total_samples = 0

    for i, image in enumerate(images):
        label = labels[i]

        embedding = classifier(image)
        pred = netC(embedding)
        pred = F.log_softmax(pred, dim=1)

        kl = split_measure(pred, label)

        total_kl += kl
        total_samples += 1


    avg_kl = total_kl/total_samples

    print ('Avg KL')
    print (avg_kl)






















