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
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device('cpu')
print(device, " will be used.\n")

train_eval = 'eval'

load_model = True

use_base_resnet = 'base'

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

    full_images = torch.zeros((1, 1, 28, 28)).detach().cpu()
    full_labels = torch.zeros((1, 10)).detach().cpu()

    num_passes = 1
    for p in range(num_passes):
        epoch_start_time = time.time()

        classifier.eval()
        netC.eval()
        netG.eval()

        for i, (data, true_label, idx) in enumerate(dataloader, 0):
            print ('Batch')
            print (i)
            real_data = data.to(device)
            true_label_g = true_label.to(device)
            bsz = real_data.shape[0]

            gen_images = torch.zeros((bsz, 1, 28, 28))
            gen_labels = torch.zeros((bsz, 10))

            embedding = classifier(real_data)
            ea = embedding.shape[0]
            eb = embedding.shape[1]
            embedding = torch.reshape(embedding, (ea, eb, 1, 1))

            for j in range(bsz):
                if (j < (bsz-1)):
                    i0 = j
                    i1 = j+1
                else:
                    i0 = j
                    i1 = 0

                diff = embedding[i1] - embedding[i0]
                diff /= 2
                noise = embedding[i0] + diff

                gen_images[i0] = netG(noise.unsqueeze(0)).detach().cpu()[0]
                l0 = true_label_g[i0]
                l1 = true_label_g[i1]
                gen_labels[i0, l0] = 0.5
                gen_labels[i0, l1] = 0.5

            full_images = torch.cat((full_images, gen_images), dim=0)
            full_labels = torch.cat((full_labels, gen_labels), dim=0)

    print ('Image Size')
    print (full_images.shape)

    print ('Labels Size')
    print (full_labels.shape)

    state = dict(
            images = full_images[1:],
            labels = full_labels[1:],
        )

    exp_dir = './checkpoints/gen_fmnist_' + str(seed)
    torch.save(state, exp_dir)
    print ('Done!')
else:
    classifier.eval()
    netC.eval()

    load_path = './checkpoints/gen_fmnist_1130'
    gen_dset = torch.load(load_path, map_location=device)

    images = gen_dset['images'].to(device)
    labels = gen_dset['labels'].to(device)

    total_kl = 0
    total_samples = 0
    test_samples = 20000

    for i, image in enumerate(images):
        label = labels[i]

        embedding = classifier(image.unsqueeze(0))
        pred = netC(embedding)
        pred = F.log_softmax(pred, dim=1)

        kl = split_measure(pred, label)

        total_kl += torch.sum(kl)
        total_samples += 1

        if (i % 1000 == 0):
            print ('Image')
            print (i)

        if (i == test_samples):
            break


    avg_kl = total_kl/total_samples

    print ('Avg KL')
    print (avg_kl)






















