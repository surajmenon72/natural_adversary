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

from models.mnist_model_wtsmooth2 import Encoder, CHead

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

load_model = False
load_classifier = False

use_base_resnet = 'base'
use_thanos_vicreg = 'thanos'
load_encoder = True

train_classifier = False
train_classifier_head = False
train_using_knn = False

load_path = ' '
state_dict = {}
if (load_model):
    load_path = './checkpoint/model_load'
    state_dict = torch.load(load_path, map_location=device)
elif (load_classifier):
    load_path = './checkpoint/model_c_load'
    state_dict = torch.load(load_path, map_location=device)

use_3_channel = False
if (use_base_resnet == 'resnet'):
    use_3_channel = True

dataloader = get_data(params['dataset'], params['batch_size'], use_3_channel=use_3_channel)
dataloader_knn = get_data(params['dataset'], params['knn_batch_size'], use_3_channel=use_3_channel)


# Initialise the network.
classifier = Encoder().to(device)
classifier.apply(weights_init)
print (classifier)

netC = CHead().to(device)
netC.apply(weights_init)
print (netC)

if (load_model):
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
    print ('Model successfully loaded')
elif (load_classifier):
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
    print ('Loaded Classifer and CHead')
else:
    #need to load classifier regardless
    if (load_encoder == True):
        if(use_thanos_vicreg == 'thanos'):
            if (use_base_resnet == 'resnet'):
                path = './checkpoints/thanos_resnet_15.ckpt'
                state_dict = torch.load(path, map_location=device)
                #missing_keys, unexpected_keys = classifier.load_state_dict(state_dict['state_dict', strict=False)
                classifier.load_state_dict(
                    {
                        ".".join(k.split(".")[3:]): v
                        for k, v in state_dict["state_dict"].items()
                        if (
                            # source_module in k
                            # and "model" in k
                            # and k.split(".")[2] == source_module
                            "model" in k
                            and "ImageEncoder" in k
                        )
                    },
                    strict=True,
                )
                print ('Loaded classifier')
            else:
                path = './checkpoints/thanos_base_20.ckpt'
                state_dict = torch.load(path, map_location=device)

                classifier.load_state_dict(
                    {
                        ".".join(k.split(".")[3:]): v
                        for k, v in state_dict["state_dict"].items()
                        if (
                            # source_module in k
                            # and "model" in k
                            # and k.split(".")[2] == source_module
                            "model" in k
                            and "ImageEncoder" in k
                        )
                    },
                    strict=True,
                )
                print ('Loaded classifier')
        else:
            #using Vicreg
            if (use_base_resnet == 'resnet'):
                path = './checkpoints/vicreg_backbone_resnet_60.pth'
                knn_path = './checkpoints/knn_vicreg_resnet.pth'
                state_dict = torch.load(path, map_location=device)

                missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)
            else:
                path = './checkpoints/vicreg_backbone_base_60.pth'
                knn_path = './checkpoints/knn_vicreg_base.pth'
                state_dict = torch.load(path, map_location=device)

                missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)

knn_path = ' '
if (train_using_knn):
    if (use_thanos_vicreg == 'thanos'):
        if (use_base_resnet == 'resnet'):
            knn_path = './checkpoints/knn_thanos_resnet.pth'
        else:
            knn_path = './checkpoints/knn_thanos_base.pth'
    else:
        if (use_base_resnet == 'resnet'):
            knn_path = './checkpoints/knn_vicreg_resnet.pth'
        else:
            knn_path = './checkpoints/knn_vicreg_base.pth'

#load knn dict regardless, assume that it matches the encoder we are using.
if (knn_path != ' '):
    #knn_path = './checkpoints/knn.pth'
    knn_dict = torch.load(knn_path)
    knn_e = knn_dict["knn_e"].to(device)
    knn_t = knn_dict["knn_t"].to(device)
    print ('Loaded KNN')

# Loss for classifier
if (train_using_knn):
    criterionC = nn.KLDivLoss()
else:
    criterionC = nn.CrossEntropyLoss()

#which networks don't require grad
if (train_classifier == False):
    classifier.requires_grad_(False)
    classifier.eval()

# Adam optimiser is used.
optimE = optim.Adam([{'params': classifier.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimC = optim.Adam([{'params': netC.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

# List variables to store results pf training.
img_list = []
C_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    total_c_loss = torch.zeros(1).to(device)
    for i, (data, true_label) in enumerate(dataloader, 0):
        # print ('Batch')
        # print (i)
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        true_label_g = true_label.to(device)

        if (train_classifier):
            classifier.train()
        Updating discriminator and DHead
        netC.train()
        optimC.zero_grad()

        #Train classifier
        #if we want to sample the knn embeddings
        if (train_classifier_head):
            if (epoch % c_train_cadence == 0):
                #print ('Training Classifier Head')
                output_c = classifier(real_data)
                probs_c = netC(output_c)
                probs_c = torch.squeeze(probs_c)
                probs_c = F.log_softmax(probs_c, dim=1)

                if (train_using_knn):
                    soft_probs_c = calculate_fuzzy_knn_eff(output_c, knn_e, knn_t, device, k=100, num_classes=10)

                # check for NaN
                isnan1 = torch.sum(torch.isnan(probs_c))
                if (train_using_knn):
                    isnan2 = torch.sum(torch.isnan(soft_probs_c))
                else:
                    isnan2 = 0

                if ((isnan1 > 0) or (isnan2 > 0)):
                    print ('NAN VALUE in Classifier Loss')

                if (train_using_knn):
                    loss_c = criterionC(probs_c, soft_probs_c)
                else:
                    loss_c = criterionC(probs_c, true_label_g)
                # Calculate gradients
                loss_c.backward()
        else:
            loss_c = torch.zeros(1)

        #Net loss for classifier
        C_loss = loss_c
        if (train_classifier):
            optimE.step()

        optimC.step()

        if (train_classifier_head):
            print ('Training Classifier Head, continuing')
            print ('C_Head Loss: %.4f\t' % (C_loss).item())
            total_c_loss += C_loss
            continue

        # Check progress of training.
        if i != 0 and i%100 == 0:
        #if i != 0 and i%10 == 0:
            print('[%d/%d][%d/%d]\tLoss_C: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    C_loss.item()))

        # Save the losses for plotting.
        C_losses.append(C_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))

    if (train_classifier_head):
        print ('C_Head Avg Loss: %.4f\t' % (total_c_loss/len(dataloader)).item())

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'classifier' : classifier.state_dict(),
            'netC' : netC.state_dict(),
            'params' : params
            }, 'checkpoint/model_calibration_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# Save network weights.
torch.save({
    'classifier' : classifier.state_dict(),
    'netC' : netC.state_dict(),
    'params' : params
    }, 'checkpoint/model_calibration_epoch_%d_{}'.format(params['dataset']) %(epoch+1))









