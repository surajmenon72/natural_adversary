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
    from models.mnist_model_wtsmooth import Generator, Discriminator, DHead, QHead, Encoder, CHead, Stretcher, HHead
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

load_model = True
load_classifier = True
train_classifier_head = False

state_dict = {}
if (load_model):
    load_path = './checkpoint/model_load'
    state_dict = torch.load(load_path, map_location=device)
elif (load_classifier):
    load_path = './checkpoint/model_c_load'
    state_dict = torch.load(load_path, map_location=device)

dataloader = get_data(params['dataset'], params['batch_size'])
dataloader_knn = get_data(params['dataset'], params['knn_batch_size'])

# Set appropriate hyperparameters depending on the dataset used.
# The values given in the InfoGAN paper are used.
# num_z : dimension of incompressible noise.
# num_dis_c : number of discrete latent code used.
# dis_c_dim : dimension of discrete latent code.
# num_con_c : number of continuous latent code used.
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    #params['num_con_c'] = 10 #continuous variable allocated for each class
    params['num_con_c'] = 2
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

netGPlus = Generator().to(device)
netG.apply(weights_init)
print (netGPlus)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print (discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead().to(device)
netQ.apply(weights_init)
print(netQ)

classifier = Encoder().to(device)
classifier.apply(weights_init)
print (classifier)

netC = CHead().to(device)
netC.apply(weights_init)
print (netC)

stretcher = Stretcher().to(device)
stretcher.apply(weights_init)
print (stretcher)

netH = HHead().to(device)
netH.apply(weights_init)
print (netH)


if (load_model):
    netG.load_state_dict(state_dict['netG'])
    netGPlus.load_state_dict(state_dict['netGPlus'])
    classifier.load_state_dict(state_dict['classifier'])
    discriminator.load_state_dict(state_dict['discriminator'])
    netD.load_state_dict(state_dict['netD'])
    netC.load_state_dict(state_dict['netC'])
    stretcher.load_state_dict(state_dict['stretcher'])
    netH.load_state_dict(state_dict['netH'])
    netQ.load_state_dict(state_dict['netQ'])
    print ('Model successfully loaded')
elif (load_classifier):
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
    print ('Loaded Classifer and CHead')
else:
    #need to load classifier regardless
    path = './checkpoints/mnist_encoder-256.pth'
    state_dict = torch.load(path, map_location=device)
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
    print ('Loaded classifier')


#load knn dict
path = './checkpoints/knn.pth'
knn_dict = torch.load(path)
knn_e = knn_dict["knn_e"].to(device)
knn_t = knn_dict["knn_t"].to(device)
print ('Loaded KNN')

# Loss for discrimination between real and fake images.
# criterionD = nn.BCELoss()
# criterionH = nn.BCELoss()
# Loss for classifier
# criterionC = nn.CrossEntropyLoss()
criterionC = nn.KLDivLoss()
# Loss for split between identity and controversy, just use CrossEntropy
criterionGP = nn.KLDivLoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

#which networks don't require grad
classifier.requires_grad_(False)
classifier.eval()

# Adam optimiser is used.
optimE = optim.Adam([{'params': classifier.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimC = optim.Adam([{'params': netC.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimGPlus = optim.Adam([{'params': netGPlus.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimS = optim.Adam([{'params': stretcher.parameters()}, {'params': netH.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

# Fixed Noise
z = torch.randn(100, params['num_z'], 1, 1, device=device)
fixed_noise = z
if(params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(10)
    dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0, 100), i, idx] = 1.0

    dis_c = dis_c.view(100, -1, 1, 1)

    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if(params['num_con_c'] != 0):
    con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

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
beta = 1
gamma = 1
clip_value_1 = 1
clip_value_2 = 1

c_train_cadence = 1
d_train_cadence = 1
gp_train_cadence = 1
gp_iters = 1
s_train_cadence = 1
g_train_cadence = 1

#save blank for first epoch
# torch.save({
#     'netG' : netG.state_dict()
#     }, 'checkpoint/gen_save')

for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data, true_label) in enumerate(dataloader, 0):
        # print ('Batch')
        # print (i)
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        true_label_g = true_label.to(device)

        #For now set G+ to G
        torch.save({
            'netG' : netG.state_dict()
            }, 'checkpoint/gen_save')

        path = './checkpoint/gen_save'
        state_dict = torch.load(path, map_location=device)
        netGPlus.load_state_dict(state_dict['netG'])


        #get labels, targets
        true_labels_hot, targets = get_targets(true_label_g, params['dis_c_dim'], device)

        #get noise sample
        noise, idx, c_nums = noise_sample_target(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device, targets)

        # Updating discriminator and DHead
        netC.train()
        optimC.zero_grad()

        #Train classifier
        #if we want to sample the knn embeddings
        if (train_classifier_head):
            if (epoch % c_train_cadence == 0):
                output_c = classifier(real_data)
                probs_c = netC(output_c)
                probs_c = torch.squeeze(probs_c)
                probs_c = F.log_softmax(probs_c, dim=1)

                knn_batches = 1
                knn_e = torch.zeros((params['knn_batch_size']*knn_batches, output_c.shape[1])).to(device)
                knn_t = torch.zeros(params['knn_batch_size']*knn_batches).to(device)
                for j, (data_knn, labels_knn) in enumerate(dataloader_knn, 0):
                    output = classifier(data_knn.to(device))
                    labels_knn = labels_knn.to(device)

                    start_index = j*params['knn_batch_size']
                    end_index = (j+1)*params['knn_batch_size']

                    knn_e[start_index:end_index, :] = output[:, :]
                    knn_t[start_index:end_index] = labels_knn[:]

                    if (j == (knn_batches-1)):
                        break

                soft_probs_c = calculate_fuzzy_knn(output_c, knn_e, knn_t, device, k=50, num_classes=10)

                # check for NaN
                isnan1 = torch.sum(torch.isnan(probs_c))
                isnan2 = torch.sum(torch.isnan(soft_probs_c))
                if ((isnan1 > 0) or (isnan2 > 0)):
                    print ('NAN VALUE in Classifier Loss')

                loss_c = criterionC(probs_c, soft_probs_c)
                loss_c = loss_c*beta
                # Calculate gradients
                loss_c.backward()
        else:
            loss_c = torch.zeros(1)

        #Net loss for classifier
        C_loss = loss_c
        optimC.step()

        netD.train()
        optimD.zero_grad()

        if (epoch % d_train_cadence == 0):
            # Real data
            real_output = discriminator(real_data)
            real_output = netD(real_output)
            errD_real = torch.mean(real_output)
            D_x = real_output.mean().item()

            # Generate fake image batch with G
            fake_data = netG(noise)

            # Train with fake
            fake_output = discriminator(fake_data.detach())
            fake_output = netD(fake_output)
            errD_fake = torch.mean(fake_output)
            D_G_z1 = fake_output.mean().item()

            # Calculate W-div gradient penalty
            # gradient_penalty = calculate_gradient_penalty(discriminator, netD,
            #                                               real_data.data, fake_data.data,
            #                                               device)
            gradient_penalty = 0

            # Add the gradients from the all-real and all-fake batches
            D_loss = -errD_real + errD_fake + gradient_penalty * 10
            D_loss.backward()
        else:
            D_loss = torch.zeros(1)

        # nn.utils.clip_grad_value_(discriminator.parameters(), clip_value_1)
        # nn.utils.clip_grad_value_(netD.parameters(), clip_value_1)
        optimD.step()
        #need to clip WGAN for Lipshitz
        clip_module_weights(discriminator, min_v=-.01, max_v=.01)
        clip_module_weights(netD, min_v=-.01, max_v=.01)

        # stretcher.train()
        # netH.train()
        # optimS.zero_grad()

        # #Train the stretcher
        # if (epoch % s_train_cadence == 0):
        #     fm = discriminator.get_feature_maps(real_data)
        #     real_output = stretcher(real_data, fm)
        #     real_output = netD(real_output)
        #     err_real = torch.mean(real_output) 

        #     fake_data_1 = netG(noise)
        #     fm = discriminator.get_feature_maps(fake_data_1)
        #     output_1 = stretcher(fake_data_1, fm)
        #     #output_1 = netH(output_1)
        #     output_1 = netD(output_1)
        #     err_g = torch.mean(output_1)

        #     fake_data_0 = netGPlus(noise)
        #     fm = discriminator.get_feature_maps(fake_data_0)
        #     output_0 = stretcher(fake_data_0, fm)
        #     #output_0 = netH(output_0)
        #     output_0 = netD(output_0)
        #     err_gplus = torch.mean(output_0)

        #     #no gradient penalty here for now
        #     g_net = -err_g + err_gplus
        #     print (g_net)
        #     S_loss = -err_real + -err_g + err_gplus
        #     # Calculate gradients.
        #     S_loss.backward()
        # else:
        #     S_loss = torch.zeros(1)
        S_loss = torch.zeros(1)

        # optimS.step()
        # #need to clip WGAN for Lipshitz
        # clip_module_weights(stretcher, min_v=-.01, max_v=.01)
        # clip_module_weights(netH, min_v=-.01, max_v=.01)

        #save generator to load next batch.. transfer netGPlus to netG
        # torch.save({
        #     'netGPlus' : netG.state_dict()
        #     }, 'checkpoint/gen_save')

        # path = './checkpoint/gen_save'
        # state_dict = torch.load(path, map_location=device)
        # netG.load_state_dict(state_dict['netGPlus'])

        # netGPlus.train()
        # optimGPlus.zero_grad()

        # #Split loss 
        # if (epoch % gp_train_cadence == 0):
        #     totalGP_loss = 0

        #     for gp_iter in range(gp_iters):
        #         split_labels = get_split_labels(true_label_g, targets, c_nums, params['dis_c_dim'], device)
        #         fake_data = netGPlus(noise)
        #         output_s = classifier(fake_data)

        #         #KLDiv expects log space, already in softmax
        #         probs_split = netC(output_s)
        #         probs_split = F.log_softmax(probs_split, dim=1)

        #         #check for NaN
        #         isnan1 = torch.sum(torch.isnan(probs_split))
        #         isnan2 = torch.sum(torch.isnan(split_labels))
        #         if ((isnan1 > 0) or (isnan2 > 0)):
        #             print ('NAN VALUE in Split Loss')

        #         loss_split = criterionGP(probs_split, split_labels)

        #         fm = discriminator.get_feature_maps(fake_data)
        #         output_h = stretcher(fake_data, fm)
        #         output_h = netH(output_h)
        #         gen_loss = torch.mean(output_h)

        #         #Loss for Split, needs to be tuned
        #         GP_loss = alpha*loss_split + beta*-gen_loss
        #         totalGP_loss += GP_loss

        #     totalGP_loss /= gp_iters
        #     totalGP_loss.backward()
        # else:
        #     totalGP_loss = torch.zeros(1)

        # optimGPlus.step()

        # # Updating Generator and QHead
        # netGPlus.train()
        # netQ.train()
        # optimGPlus.zero_grad()

        netG.train()
        netQ.train()
        optimG.zero_grad()
        totalGP_loss = torch.zeros(1)

        # Fake data treated as real.
        if (epoch % g_train_cadence == 0):
             # Now set the latent var
            # fake_data = netGPlus(noise)
            # output_q = discriminator(fake_data)

            fake_data = netG(noise)
            output_d = discriminator(fake_data)
            output_d = netD(output_d)
            err_d = torch.mean(output_d) 

            fm = discriminator.get_feature_maps(fake_data)
            output_s = stretcher(fake_data, fm)
            #output_s = netH(output_s)
            output_s = netD(output_s)
            err_s = torch.mean(output_s)

            output_q = discriminator(fake_data)
            q_logits, q_mu, q_var = netQ(output_q)
            target = torch.LongTensor(idx).to(device)
            # Calculating loss for discrete latent code.
            dis_loss = 0

            isnan1 = torch.sum(torch.isnan(q_logits))
            isnan2 = torch.sum(torch.isnan(target))
            if ((isnan1 > 0) or (isnan2 > 0)):
                print ('NAN VALUE in Q Discrete Loss')

            # for MNIST, this is always 1
            for j in range(params['num_dis_c']):
                dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

            # align_loss = criterionQ_dis(q_logits, true_label_g)
            # align_loss = 0

            isnan1 = torch.sum(torch.isnan(noise))
            isnan2 = torch.sum(torch.isnan(q_mu))
            isnan3 = torch.sum(torch.isnan(q_var))
            if ((isnan1 > 0) or (isnan2 > 0) or (isnan3 > 0)):
                print ('NAN VALUE in Q Continuous Loss')

            # Calculating loss for continuous latent code.
            con_loss = 0
            if (params['num_con_c'] != 0):
                con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1

            # Net loss for generator.
            #G_loss = torch.zeros(1)
            #G_loss = -err_d + -err_s*gamma
            G_loss = -err_d
            Q_loss = dis_loss + con_loss
            #GQ_loss = G_loss + Q_loss
            GQ_loss = G_loss + Q_loss
            # Calculate gradients.
            GQ_loss.backward()
        else:
            G_loss = torch.zeros(1)
            Q_loss = torch.zeros(1)

        #optimGPlus.step()
        optimG.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
        #if i != 0 and i%10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tLoss_S: %.4f\tLoss_Q: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item(), C_loss.item(), S_loss.item(), Q_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        Q_losses.append(Q_loss.item())
        D_losses.append(D_loss.item())
        C_losses.append(C_loss.item())
        GP_losses.append(totalGP_loss.item())
        S_losses.append(S_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
        plt.close('all')

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'netGPlus' : netGPlus.state_dict(),
            'classifier' : classifier.state_dict(),
            'discriminator': discriminator.state_dict(),
            'stretcher' : stretcher.state_dict(),
            'netC' : netC.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'netH' : netH.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'optimS' : optimS.state_dict(),
            'optimGPlus' : optimGPlus.state_dict(),
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
    'netGPlus' : netGPlus.state_dict(),
    'classifier' : classifier.state_dict(),
    'discriminator': discriminator.state_dict(),
    'stretcher' : stretcher.state_dict(),
    'netC' : netC.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'netH' : netH.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'optimS' : optimS.state_dict(),
    'optimGPlus' : optimGPlus.state_dict(),
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