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

from models.mnist_model_wtsmooth2 import Encoder, CHead, CheapClassifier

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

train_eval = 'train'

load_model = False
load_ensemble = False

use_base_resnet = 'base'
use_thanos_vicreg = 'vicreg'
load_encoder = True

train_classifier = False
train_classifier_head = True
train_using_knn = True
train_ensemble = False

load_path = ' '
state_dict = {}
if (load_model):
    load_path = './checkpoint/model_c_load'
    state_dict = torch.load(load_path, map_location=device)

ensemble_path = ' '
ensemble_dict = {}
if (load_ensemble):
    load_path = './checkpoint/model_e_load'
    ensemble_dict = torch.load(load_path, map_location=device)

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

#Ensemble of Cheap Cs
num_ccs = 10
cc0 = CheapClassifier().to(device)
cc0.apply(weights_init)
cc1 = CheapClassifier().to(device)
cc1.apply(weights_init)
cc2 = CheapClassifier().to(device)
cc2.apply(weights_init)
cc3 = CheapClassifier().to(device)
cc3.apply(weights_init)
cc4 = CheapClassifier().to(device)
cc4.apply(weights_init)
cc5 = CheapClassifier().to(device)
cc5.apply(weights_init)
cc6 = CheapClassifier().to(device)
cc6.apply(weights_init)
cc7 = CheapClassifier().to(device)
cc7.apply(weights_init)
cc8 = CheapClassifier().to(device)
cc8.apply(weights_init)
cc9 = CheapClassifier().to(device)
cc9.apply(weights_init)
ccm = CheapClassifier().to(device)
ccm.apply(weights_init)

if (load_model):
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
    print ('Model successfully loaded')
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
                path = './checkpoints/thanos_base_fashion_30.ckpt'
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
                state_dict = torch.load(path, map_location=device)

                missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)
            else:
                #path = './checkpoints/vicreg_backbone_base_fashion_60.pth'
                path = './checkpoints/supvic_backbone_base_fashion_60.pth'
                state_dict = torch.load(path, map_location=device)

                missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)

if (load_ensemble == True):
    cc0.load_state_dict(ensemble_dict['cc0'])
    cc1.load_state_dict(ensemble_dict['cc1'])
    cc2.load_state_dict(ensemble_dict['cc2'])
    cc3.load_state_dict(ensemble_dict['cc3'])
    cc4.load_state_dict(ensemble_dict['cc4'])
    cc5.load_state_dict(ensemble_dict['cc5'])
    cc6.load_state_dict(ensemble_dict['cc6'])
    cc7.load_state_dict(ensemble_dict['cc7'])
    cc8.load_state_dict(ensemble_dict['cc8'])
    cc9.load_state_dict(ensemble_dict['cc9'])
    ccm.load_state_dict(ensemble_dict['ccm'])
    print ('Loaded Ensemble')


knn_path = ' '
if (train_using_knn):
    if (use_thanos_vicreg == 'thanos'):
        if (use_base_resnet == 'resnet'):
            knn_path = './checkpoints/knn_thanos_resnet.pth'
        else:
            knn_path = './checkpoints/knn_thanos_base_fashion.pth'
    else:
        if (use_base_resnet == 'resnet'):
            knn_path = './checkpoints/knn_vicreg_resnet.pth'
        else:
            #knn_path = './checkpoints/knn_vicreg_base_fashion.pth'
            knn_path = './checkpoints/knn_supvic_base_fashion.pth'

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

criterionCC = nn.CrossEntropyLoss()

#which networks don't require grad
if (train_classifier == False):
    classifier.requires_grad_(False)
    classifier.eval()

# Adam optimiser is used.
optimE = optim.Adam([{'params': classifier.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimC = optim.Adam([{'params': netC.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimcc = optim.Adam([{'params': cc0.parameters()}, 
                      {'params': cc1.parameters()},
                      {'params': cc2.parameters()},
                      {'params': cc3.parameters()},
                      {'params': cc4.parameters()},
                      {'params': cc5.parameters()},
                      {'params': cc6.parameters()},
                      {'params': cc7.parameters()},
                      {'params': cc8.parameters()},
                      {'params': cc9.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimccm = optim.Adam([{'params': ccm.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

# List variables to store results pf training.
img_list = []
C_losses = []
E_losses = []
M_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0
num_batches = len(dataloader)
cc_iter = int(num_batches/num_ccs)

if (train_eval == 'train'):
    for epoch in range(params['num_epochs']):
        epoch_start_time = time.time()

        total_c_loss = torch.zeros(1).to(device)
        for i, (data, true_label, idx) in enumerate(dataloader, 0):
            # print ('Batch')
            # print (i)
            # Get batch size
            b_size = data.size(0)
            # Transfer data tensor to GPU/CPU (device)
            real_data = data.to(device)
            true_label_g = true_label.to(device)

            #Train classifier
            #if we want to sample the knn embeddings
            if (train_classifier_head):
                if (train_classifier):
                    classifier.train()
                
                netC.train()
                optimC.zero_grad()
                
                #print ('Training Classifier Head')
                output_c = classifier(real_data)
                probs_c = netC(output_c)
                probs_c = torch.squeeze(probs_c)
                if (train_using_knn):
                    probs_c = F.log_softmax(probs_c, dim=1)

                if (train_using_knn):
                    soft_probs_c = calculate_fuzzy_knn_eff(output_c, knn_e, knn_t, device, k=10, num_classes=10)

                if (train_using_knn):
                    loss_c = criterionC(probs_c, soft_probs_c)
                else:
                    loss_c = criterionC(probs_c, true_label_g)
                # Calculate gradients
                loss_c.backward()

                if (train_classifier):
                    optimE.step()

                optimC.step()
            else:
                loss_c = torch.zeros(1)

            C_loss = loss_c

            if (train_ensemble):
                cc0.train()
                cc1.train()
                cc2.train()
                cc3.train()
                cc4.train()
                cc5.train()
                cc6.train()
                cc7.train()
                cc8.train()
                cc9.train()
                ccm.train()
                optimcc.zero_grad()
                optimccm.zero_grad()

                if (i < cc_iter):
                    output_cc = cc0(real_data)
                elif (i >= cc_iter and i < (cc_iter*2)):
                    output_cc = cc1(real_data)
                elif (i >= (cc_iter*2) and i < (cc_iter*3)):
                    output_cc = cc2(real_data)
                elif (i >= (cc_iter*3) and i < (cc_iter*4)):
                    output_cc = cc3(real_data)
                elif (i >= (cc_iter*4) and i < (cc_iter*5)):
                    output_cc = cc4(real_data)
                elif (i >= (cc_iter*5) and i < (cc_iter*6)):
                    output_cc = cc5(real_data)
                elif (i >= (cc_iter*6) and i < (cc_iter*7)):
                    output_cc = cc6(real_data)
                elif (i >= (cc_iter*7) and i < (cc_iter*8)):
                    output_cc = cc7(real_data)
                elif (i >= (cc_iter*8) and i < (cc_iter*9)):
                    output_cc = cc8(real_data)
                else:
                    output_cc = cc9(real_data)

                output_ccm = ccm(real_data)

                probs_cc = torch.squeeze(output_cc)
                #probs_cc = F.log_softmax(probs_cc, dim=1)

                loss_cc = criterionCC(probs_cc, true_label_g)
                loss_cc.backward()
                optimcc.step()

                probs_ccm = torch.squeeze(output_ccm)
                #probs_ccm = F.log_softmax(probs_ccm, dim=1)

                loss_ccm = criterionCC(probs_ccm, true_label_g)
                loss_ccm.backward()
                optimccm.step()
            else:
                loss_cc = torch.zeros(1)
                loss_ccm = torch.zeros(1)

            E_loss = loss_cc
            M_loss = loss_ccm

            # Check progress of training.
            if i != 0 and i%100 == 0:
            #if i != 0 and i%10 == 0:
                print('[%d/%d][%d/%d]\tLoss_C: %.4f\t Loss E: %.4f\t Loss_M: %.4f'
                      % (epoch+1, params['num_epochs'], i, len(dataloader), 
                        C_loss.item(), E_loss.item(), M_loss.item()))

            # Save the losses for plotting.
            C_losses.append(C_loss.item())
            E_losses.append(E_loss.item())
            M_losses.append(M_loss.item())

            iters += 1

        epoch_time = time.time() - epoch_start_time
        print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))

        # Save network weights.
        if (epoch+1) % params['save_epoch'] == 0:
            torch.save({
                'classifier' : classifier.state_dict(),
                'netC' : netC.state_dict(),
                'params' : params
                }, 'checkpoint/model_calibration_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

        if (epoch+1) % params['save_epoch'] == 0:
            torch.save({
                'cc0' : cc0.state_dict(),
                'cc1' : cc1.state_dict(),
                'cc2' : cc2.state_dict(),
                'cc3' : cc3.state_dict(),
                'cc4' : cc4.state_dict(),
                'cc5' : cc5.state_dict(),
                'cc6' : cc6.state_dict(),
                'cc7' : cc7.state_dict(),
                'cc8' : cc8.state_dict(),
                'cc9' : cc9.state_dict(),
                'ccm' : ccm.state_dict(),
                'params' : params
                }, 'checkpoint/model_ensemble_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

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

    torch.save({
        'cc0' : cc0.state_dict(),
        'cc1' : cc1.state_dict(),
        'cc2' : cc2.state_dict(),
        'cc3' : cc3.state_dict(),
        'cc4' : cc4.state_dict(),
        'cc5' : cc5.state_dict(),
        'cc6' : cc6.state_dict(),
        'cc7' : cc7.state_dict(),
        'cc8' : cc8.state_dict(),
        'cc9' : cc9.state_dict(),
        'ccm' : ccm.state_dict(),
        'params' : params
        }, 'checkpoint/model_ensemble_epoch_%d_{}'.format(params['dataset']) %(epoch+1))
else:
    classifier.eval()
    netC.eval()
    cc0.eval()
    cc1.eval()
    cc2.eval()
    cc3.eval()
    cc4.eval()
    cc5.eval()
    cc6.eval()
    cc7.eval()
    cc8.eval()
    cc9.eval()
    ccm.eval()

    total_correct_c = 0
    total_correct_avg = 0
    total_correct_ccm = 0
    total_entropy_c = 0
    total_entropy_avg = 0
    total_entropy_ccm = 0
    total_samples = 0
    for i, (data, true_label, idx) in enumerate(dataloader_eval, 0):
        # print ('Batch')
        # print (i)
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        true_label_g = true_label.to(device)

        output_c = classifier(real_data)
        probs_c = netC(output_c)
        probs_c = F.softmax(torch.squeeze(probs_c), dim=1)

        output_cc0 = cc0(real_data)
        output_cc1 = cc1(real_data)
        output_cc2 = cc2(real_data)
        output_cc3 = cc3(real_data)
        output_cc4 = cc4(real_data)
        output_cc5 = cc5(real_data)
        output_cc6 = cc6(real_data)
        output_cc7 = cc7(real_data)
        output_cc8 = cc8(real_data)
        output_cc9 = cc9(real_data)
        output_ccm = ccm(real_data)

        probs_cc0 = F.softmax(torch.squeeze(output_cc0), dim=1)
        probs_cc1 = F.softmax(torch.squeeze(output_cc1), dim=1)
        probs_cc2 = F.softmax(torch.squeeze(output_cc2), dim=1)
        probs_cc3 = F.softmax(torch.squeeze(output_cc3), dim=1)
        probs_cc4 = F.softmax(torch.squeeze(output_cc4), dim=1)
        probs_cc5 = F.softmax(torch.squeeze(output_cc5), dim=1)
        probs_cc6 = F.softmax(torch.squeeze(output_cc6), dim=1)
        probs_cc7 = F.softmax(torch.squeeze(output_cc7), dim=1)
        probs_cc8 = F.softmax(torch.squeeze(output_cc8), dim=1)
        probs_cc9 = F.softmax(torch.squeeze(output_cc9), dim=1)

        probs_sum = (probs_cc0 + probs_cc1 + probs_cc2 + probs_cc3 + probs_cc4 +
                     probs_cc5 + probs_cc6 + probs_cc7 + probs_cc8 + probs_cc9)

        probs_avg = probs_sum/10
        probs_avg = probs_avg/(torch.sum(probs_avg, dim=1).view((probs_avg.shape[0], 1)))
        #probs_avg = F.softmax(probs_avg, dim=1)

        probs_ccm = F.softmax(torch.squeeze(output_ccm), dim=1)

        if (i % 10 == 0):
            print ('Probs KNN:')
            print (probs_c[0])
            print ('Probs AVG:')
            print (probs_avg[0])
            print ('Probs Master:')
            print (probs_ccm[0])

        guess_c = torch.argmax(probs_c, dim=1)
        guess_avg = torch.argmax(probs_avg, dim=1)
        guess_ccm = torch.argmax(probs_ccm, dim=1)
     
        equal_c = (true_label_g == guess_c)
        equal_avg = (true_label_g == guess_avg)
        equal_ccm = (true_label_g == guess_ccm)

        total_correct_c += torch.sum(equal_c)
        total_correct_avg += torch.sum(equal_avg)
        total_correct_ccm += torch.sum(equal_ccm)
        total_samples += b_size

        c_entropy = calc_entropy(probs_c)
        avg_entropy = calc_entropy(probs_avg)
        ccm_entropy = calc_entropy(probs_ccm)

        total_entropy_c += torch.sum(c_entropy)
        total_entropy_avg += torch.sum(avg_entropy)
        total_entropy_ccm += torch.sum(ccm_entropy)


    #Accuracies
    accuracy_c = (total_correct_c/total_samples)
    accuracy_avg = (total_correct_avg/total_samples)
    accuracy_ccm = (total_correct_ccm/total_samples)

    #Entropies
    e_c = (total_entropy_c/total_samples)
    e_avg = (total_entropy_avg/total_samples)
    e_ccm = (total_entropy_ccm/total_samples)

    print ('Accuracy KNN C: %.4f' % (accuracy_c))
    print ('Accuracy AVG C: %.4f' % (accuracy_avg))
    print ('Accuracy CCM C: %.4f' % (accuracy_ccm))

    print ('Entropy KNN C: %.4f' % (e_c))
    print ('Entropy AVG C: %.4f' % (e_avg))
    print ('Entropy CCM C: %.4f' % (e_ccm))












