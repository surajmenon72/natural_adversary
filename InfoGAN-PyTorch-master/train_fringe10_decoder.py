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

#from models.mnist_model_exp import Generator, Discriminator, DHead, Classifier, CHead, SHead, QHead
from dataloader import get_data
from utils import *
from config import params

if(params['dataset'] == 'MNIST'):
    from models.mnist_model_wtsmooth2 import Generator, Discriminator, DHead, DHead_KL, QHead, Encoder, ResnetEncoder, CHead, Stretcher, HHead
elif(params['dataset'] == 'SVHN'):
    from models.svhn_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead

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
use_thanos_vicreg = 'vicreg'
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

#dataloader = get_data(params['dataset'], params['batch_size'], use_3_channel=use_3_channel)
dataloader = get_data('FashionMNIST', params['batch_size'], train_test='train_index', use_3_channel=use_3_channel)
dataloader_knn = get_data(params['dataset'], params['knn_batch_size'], use_3_channel=use_3_channel)

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 256
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    #params['num_con_c'] = 10 #continuous variable allocated for each class
    params['num_con_c'] = 1
elif(params['dataset'] == 'SVHN'):
    params['num_z'] = 124
    params['num_dis_c'] = 4
    params['dis_c_dim'] = 10
    params['num_con_c'] = 4
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0
elif(params['dataset'] == 'FashionMNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('Training Images {}'.format(params['dataset']))
plt.close('all')

# Initialise the network.
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print (discriminator)

netD = DHead_KL().to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead().to(device)
netQ.apply(weights_init)
print(netQ)

#classifier = ResnetEncoder().to(device)
classifier = Encoder().to(device)
classifier.apply(weights_init)
print (classifier)

netC = CHead().to(device)
netC.apply(weights_init)
print (netC)

if (load_model):
    netG.load_state_dict(state_dict['netG'])
    classifier.load_state_dict(state_dict['classifier'])
    discriminator.load_state_dict(state_dict['discriminator'])
    netD.load_state_dict(state_dict['netD'])
    netC.load_state_dict(state_dict['netC'])
    netQ.load_state_dict(state_dict['netQ'])
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

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for classifier
if (train_using_knn):
    criterionC = nn.KLDivLoss()
else:
    criterionC = nn.CrossEntropyLoss()
# Loss for split between identity and controversy, just use CrossEntropy
criterionG = nn.KLDivLoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

criterionDecode = nn.MSELoss()
#criterionRecon = nn.BCELoss(reduction='mean')
#criterionRecon = nn.MSELoss()
criterionRecon = nn.BCEWithLogitsLoss(reduction='mean')

#which networks don't require grad
if (train_classifier == False):
    classifier.requires_grad_(False)
    classifier.eval()

# Adam optimiser is used.
optimE = optim.Adam([{'params': classifier.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimC = optim.Adam([{'params': netC.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

#Fixed Noise
#z = torch.randn(100, params['num_z'], 1, 1, device=device)
z = torch.rand(100, params['num_z'], 1, 1, device=device)
start_noise = z
fixed_noise = torch.zeros(100, params['num_z'], 1, 1, device=device)

#trying a spectrum test
for s in range(10):
    index = 10*s
    fixed_noise[index] = start_noise[index]
    fixed_noise[index+9] = start_noise[index+1]
    diff = start_noise[index+1] - start_noise[index]
    diff = diff/8
    for sp in range(1, 9):
        fixed_noise[index+sp] = fixed_noise[index+sp-1] + diff
# if(params['num_dis_c'] != 0):
#     idx = np.arange(params['dis_c_dim']).repeat(10)
#     dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
#     for i in range(params['num_dis_c']):
#         dis_c[torch.arange(0, 100), i, idx] = 1.0

#     dis_c = dis_c.view(100, -1, 1, 1)

#     fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

# if(params['num_con_c'] != 0):
#     con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
#     fixed_noise = torch.cat((fixed_noise, con_c), dim=1)


#only if using a binary loss for fake/real
real_label = 1
fake_label = 0

# List variables to store results pf training.
img_list = []
G_losses = []
GP_losses = []
Q_losses = []
D_losses = []
C_losses = []
S_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0

#Realness vs. Classification Hyperparams
alpha = 1
gamma = 1
beta = 1
clip_value_1 = 1
clip_grads = False

c_train_cadence = 1
d_train_cadence = 1
g_train_cadence = 1

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


        #get labels, targets for split
        true_labels_hot, targets = get_targets(true_label_g, params['dis_c_dim'], device)

        #get noise sample
        noise, idx, c_nums = noise_sample_target(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device, targets, dist='Uniform')
        z_noise = noise[:, :params['num_z']]
        # if (train_classifier):
        #     classifier.train()
        # Updating discriminator and DHead
        # netC.train()
        # optimC.zero_grad()

        # #Train classifier
        # #if we want to sample the knn embeddings
        # if (train_classifier_head):
        #     if (epoch % c_train_cadence == 0):
        #         #print ('Training Classifier Head')
        #         output_c = classifier(real_data)
        #         probs_c = netC(output_c)
        #         probs_c = torch.squeeze(probs_c)
        #         probs_c = F.log_softmax(probs_c, dim=1)

        #         if (train_using_knn):
        #             soft_probs_c = calculate_fuzzy_knn_eff(output_c, knn_e, knn_t, device, k=100, num_classes=10)

        #         # check for NaN
        #         isnan1 = torch.sum(torch.isnan(probs_c))
        #         if (train_using_knn):
        #             isnan2 = torch.sum(torch.isnan(soft_probs_c))
        #         else:
        #             isnan2 = 0

        #         if ((isnan1 > 0) or (isnan2 > 0)):
        #             print ('NAN VALUE in Classifier Loss')

        #         if (train_using_knn):
        #             loss_c = criterionC(probs_c, soft_probs_c)
        #         else:
        #             loss_c = criterionC(probs_c, true_label_g)
        #         # Calculate gradients
        #         loss_c.backward()
        # else:
        #     loss_c = torch.zeros(1)

        # #Net loss for classifier
        # C_loss = loss_c
        # if (train_classifier):
        #     optimE.step()

        # optimC.step()

        # if (train_classifier_head):
        #     print ('Training Classifier Head, continuing')
        #     print ('C_Head Loss: %.4f\t' % (C_loss).item())
        #     total_c_loss += C_loss
        #     continue

        netD.train()
        optimD.zero_grad()

        if (epoch % d_train_cadence == 0):
            # Real data
            label = torch.full((b_size, ), real_label, device=device)
            real_output = discriminator(real_data)
            probs_real = netD(real_output).view(-1)
            label = label.to(torch.float32)
            loss_real = criterionD(probs_real, label)
            #calculate grad
            loss_real.backward()

            # Generate fake image batch with G
            fake_data = netG(z_noise)
            #fake_data = torch.cat([fake_data, fake_data, fake_data], dim=1) 

            # Train with fake
            label.fill_(fake_label)
            fake_output = discriminator(fake_data.detach())
            probs_fake = netD(fake_output).view(-1)
            label = label.to(torch.float32)
            loss_fake = criterionD(probs_fake, label)
            #calculate grad
            loss_fake.backward()

            D_loss = loss_real + loss_fake
        else:
            D_loss = torch.zeros(1)

        # if (clip_grads):
        #     nn.utils.clip_grad_value_(discriminator.parameters(), clip_value_1)
        #     nn.utils.clip_grad_value_(netD.parameters(), clip_value_1)
        optimD.step()

        netG.train()
        #netQ.train()
        optimG.zero_grad()

        #Split loss 
        if (epoch % g_train_cadence == 0):
            totalG_loss = 0
            total_dec_loss = 0
            total_gen_d_loss = 0

            # embedding = classifier(real_data)
            # ea = embedding.shape[0]
            # eb = embedding.shape[1]
            # embedding = torch.reshape(embedding, (ea, eb, 1, 1))

            #test using the real embedding
            #fixed_noise = embedding[:100]

            #fixed noise from distribution
            # fake_data = torch.randn(b_size, 1, 28, 28, device=device)
            # fake_embedding = classifier(real_data)
            # fake_embedding = torch.reshape(fake_embedding, (ea, eb, 1, 1))
            # fixed_noise = fake_embedding[:100]

 
            #print (embedding[0])

            # isnan = torch.sum(torch.isnan(embedding))
            # print ('Is Embedding NAN')
            # print (isnan)

            #split_labels = get_split_labels(true_label_g, targets, c_nums, params['dis_c_dim'], device)
            #fake_data = netG(z_noise)
            #reconstruction = netG(embedding)
            #reconstruction = netG.f_logits(embedding)

            #print (reconstruction[0])

            # isnan = torch.sum(torch.isnan(reconstruction))
            # print ('Is Reconstruction NAN')
            # print (isnan)

            #print (real_data.shape)
            #print (reconstruction.shape)

            #reconstruction_loss = criterionRecon(reconstruction, real_data)
            #print (reconstruction_loss)

            # if (use_3_channel):
            #     fake_data = torch.cat([fake_data, fake_data, fake_data], dim=1)

            # output_s = classifier(fake_data)

            # z_noise_s = torch.squeeze(z_noise)
            # dec_loss = criterionDecode(output_s, z_noise_s)


            #KLDiv expects log space, already in softmax
            # probs_split = netC(output_s)
            # probs_split = F.log_softmax(probs_split, dim=1)

            # #check for NaN
            # isnan1 = torch.sum(torch.isnan(probs_split))
            # isnan2 = torch.sum(torch.isnan(split_labels))
            # if ((isnan1 > 0) or (isnan2 > 0)):
            #     print ('NAN VALUE in Split Loss')

            # loss_split = criterionG(probs_split, split_labels)


            label = torch.full((b_size, ), real_label, device=device)
            fake_data = netG(z_noise)
            output_d = discriminator(fake_data)
            probs_fake = netD(output_d).view(-1)
            label = label.to(torch.float32)
            gen_d_loss = criterionD(probs_fake, label)

            #Loss for Split, needs to be tuned
            #G_loss = alpha*loss_split + gamma*gen_d_loss
            #G_loss = alpha*dec_loss + gamma*gen_d_loss
            G_loss = gen_d_loss
            #G_loss = reconstruction_loss
            totalG_loss += G_loss
            
            #total_dec_loss += dec_loss
            #total_gen_d_loss += gen_d_loss

            G_loss.backward()

            # total_grad = 0
            # for j, p in enumerate(netG.parameters()):
            #     #print (p.grad.norm())
            #     print (p.grad.norm())
            #     total_grad += p.grad.norm()

            # print (total_grad)
            # exit()

            # fake_data = netG(noise)
            # #fake_data = torch.cat([fake_data, fake_data, fake_data], dim=1) 

            # output_q = discriminator(fake_data)
            # q_logits, q_mu, q_var = netQ(output_q)
            # target = torch.LongTensor(idx).to(device)
            # # Calculating loss for discrete latent code.
            # dis_loss = 0

            # isnan1 = torch.sum(torch.isnan(q_logits))
            # isnan2 = torch.sum(torch.isnan(target))
            # if ((isnan1 > 0) or (isnan2 > 0)):
            #     print ('NAN VALUE in Q Discrete Loss')

            # # for MNIST, this is always 1
            # for j in range(params['num_dis_c']):
            #     dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

            # # align_loss = criterionQ_dis(q_logits, true_label_g)
            # # align_loss = 0

            # isnan1 = torch.sum(torch.isnan(noise))
            # isnan2 = torch.sum(torch.isnan(q_mu))
            # isnan3 = torch.sum(torch.isnan(q_var))
            # if ((isnan1 > 0) or (isnan2 > 0) or (isnan3 > 0)):
            #     print ('NAN VALUE in Q Continuous Loss')

            # # Calculating loss for continuous latent code.
            # con_loss = 0
            # if (params['num_con_c'] != 0):
            #     con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1

            # Q_loss = dis_loss + con_loss
            # Q_loss = beta*Q_loss
            # Q_loss.backward()

        else:
            totalG_loss = torch.zeros(1)

        #nn.utils.clip_grad_value_(netG.parameters(), clip_value_1)
        optimG.step()

        Q_loss = torch.zeros(1)
        #D_loss = torch.zeros(1)
        C_loss = torch.zeros(1)
        total_dec_loss = torch.zeros(1)
        total_gen_d_loss = torch.zeros(1)

        # Check progress of training.
        if i != 0 and i%100 == 0:
        #if i != 0 and i%10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tLoss_Q: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item(), C_loss.item(), Q_loss.item()))
            print('[%d/%d][%d/%d]\tLoss_Dec: %.4f\tLoss_Gen_D: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    total_dec_loss.item(), total_gen_d_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        Q_losses.append(Q_loss.item())
        D_losses.append(D_loss.item())
        C_losses.append(C_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    #if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
    if (epoch+1) % params['save_epoch'] == 0:
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
        plt.close('all')

    if (train_classifier_head):
        print ('C_Head Avg Loss: %.4f\t' % (total_c_loss/len(dataloader)).item())

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'classifier' : classifier.state_dict(),
            'discriminator': discriminator.state_dict(),
            'netC' : netC.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'optimC' : optimC.state_dict(),
            'optimE' : optimE.state_dict(),
            'params' : params
            }, 'checkpoint/model_fringe_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# Generate image to check performance of trained generator.
with torch.no_grad():
    gen_data = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("Epoch_%d_{}".format(params['dataset']) %(params['num_epochs']))

# Save network weights.
torch.save({
    'netG' : netG.state_dict(),
    'classifier' : classifier.state_dict(),
    'discriminator': discriminator.state_dict(),
    'netC' : netC.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'optimC' : optimC.state_dict(),
    'optimE' : optimE.state_dict(),
    'params' : params
    }, 'checkpoint/model_fringe_epoch_%d_{}'.format(params['dataset']) %(epoch+1))


# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss Curve {}".format(params['dataset']))

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(10,10))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
plt.show()