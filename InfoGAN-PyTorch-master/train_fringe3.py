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

from models.mnist_model_exp import Generator, Discriminator, DHead, Classifier, CHead, SHead, QHead
from dataloader import get_data
from utils import *
from config import params

if(params['dataset'] == 'MNIST'):
    from models.mnist_model_smooth import Generator, Discriminator, DHead, Encoder, CHead, QHead
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
load_classifier = True
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
    params['num_con_c'] = 10 #continuous variable allocated for each class
    #params['num_con_c'] = 1
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
print(discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

classifier = Encoder().to(device)
classifier.apply(weights_init)
print(classifier)

netC = CHead().to(device)
netC.apply(weights_init)
print (netC)

netQ = QHead().to(device)
netQ.apply(weights_init)
print(netQ)


if (load_model):
    netG.load_state_dict(state_dict['netG'])
    discriminator.load_state_dict(state_dict['discriminator'])
    netD.load_state_dict(state_dict['netD'])
    classifier.load_state_dict(state_dict['classifier'])
    netC.load_state_dict(state_dict['netC'])
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
    missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)
    print ('Loaded classifier')


#load knn dict
path = './checkpoints/knn.pth'
knn_dict = torch.load(path)
knn_e = knn_dict["knn_e"].to(device)
knn_t = knn_dict["knn_t"].to(device)
print ('Loaded KNN')

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for classifier
# criterionC = nn.CrossEntropyLoss()
criterionC = nn.KLDivLoss()
# Loss for split between identity and controversy, just use CrossEntropy
criterionS = nn.KLDivLoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

#which networks don't require grad
classifier.requires_grad_(False)
classifier.eval()

# Adam optimiser is used.
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}, {'params': classifier.parameters()}, {'params': netC.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

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

real_label = 1
fake_label = 0

# List variables to store results pf training.
img_list = []
G_losses = []
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
d_loose = 1
e_loose = .5
clip_value_1 = 1
clip_value_2 = 1

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


        #get labels, targets
        true_labels_hot, targets = get_targets(true_label_g, params['dis_c_dim'], device)

        #get noise sample
        noise, idx, c_nums = noise_sample_target(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device, targets)

        # Updating discriminator and DHead
        discriminator.train()
        netD.train()
        #classifier.train()
        netC.train()
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size, ), real_label, device=device)
        output1 = discriminator(real_data)
        probs_real = netD(output1).view(-1)
        label = label.to(torch.float32)

        #check for NaN
        isnan1 = torch.sum(torch.isnan(probs_real))
        isnan2 = torch.sum(torch.isnan(label))
        if ((isnan1 > 0) or (isnan2 > 0)):
            print ('NAN VALUE in Discriminator Real Loss')

        loss_real = criterionD(probs_real, label)
        loss_real = loss_real*alpha
        # Calculate gradients.
        loss_real.backward()

        #Train classifier
        # output_c = classifier(real_data)
        # probs_c = netC(output_c)
        # probs_c = torch.squeeze(probs_c)
        # probs_c = F.log_softmax(probs_c, dim=1)

        # if we want to sample the knn embeddings
        # knn_batches = 1
        # knn_e = torch.zeros((params['knn_batch_size']*knn_batches, output_s.shape[1])).to(device)
        # knn_t = torch.zeros(params['knn_batch_size']*knn_batches).to(device)
        # for j, (data_knn, labels_knn) in enumerate(dataloader_knn, 0):
        #     output = classifier(data_knn.to(device))
        #     labels_knn = labels_knn.to(device)

        #     start_index = j*params['knn_batch_size']
        #     end_index = (j+1)*params['knn_batch_size']

        #     knn_e[start_index:end_index, :] = output[:, :]
        #     knn_t[start_index:end_index] = labels_knn[:]

        #     if (j == (knn_batches-1)):
        #         break

        # soft_probs_c = calculate_fuzzy_knn(output_c, knn_e, knn_t, device, k=50, num_classes=10)

        #check for NaN
        # isnan1 = torch.sum(torch.isnan(probs_c))
        # isnan2 = torch.sum(torch.isnan(soft_probs_c))
        # if ((isnan1 > 0) or (isnan2 > 0)):
        #     print ('NAN VALUE in Classifier Loss')

        # loss_c = criterionC(probs_c, soft_probs_c)
        # loss_c = loss_c*beta
        # # Calculate gradients
        # loss_c.backward()
        loss_c = torch.zeros(1)

        # Fake data
        label.fill_(fake_label)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)

        isnan1 = torch.sum(torch.isnan(probs_fake))
        isnan2 = torch.sum(torch.isnan(label))
        if ((isnan1 > 0) or (isnan2 > 0)):
            print ('NAN VALUE in Discriminator Fake Loss')

        loss_fake = criterionD(probs_fake, label)
        loss_fake = loss_fake*alpha*d_loose
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator and classifier
        D_loss = loss_real + loss_fake

        # Update parameters
        nn.utils.clip_grad_value_(discriminator.parameters(), clip_value_1)
        nn.utils.clip_grad_value_(netD.parameters(), clip_value_1)
        # nn.utils.clip_grad_value_(classifier.parameters(), clip_value_1)
        nn.utils.clip_grad_value_(netC.parameters(), clip_value_1)
        optimD.step()

        # Updating Generator and QHead
        netG.train()
        netQ.train()
        optimG.zero_grad()
        optimD.zero_grad() 

        #Split loss 
        # split_labels = get_split_labels(true_label_g, targets, c_nums, params['dis_c_dim'], device)
        # fake_data = netG(noise)
        # output_s = classifier(fake_data)

        # #KLDiv expects log space, already in softmax
        # probs_split = netC(output_s)
        # probs_split = F.log_softmax(probs_split, dim=1)

        # #check for NaN
        # isnan1 = torch.sum(torch.isnan(probs_split))
        # isnan2 = torch.sum(torch.isnan(split_labels))
        # if ((isnan1 > 0) or (isnan2 > 0)):
        #     print ('NAN VALUE in Split Loss')

        # loss_split = criterionS(probs_split, split_labels)
        # loss_split = loss_split*beta
        # #Calculate Gradients
        # loss_split.backward()
        # loss_split = torch.zeros(1)

        fake_data = netG(noise)
        output_e = classifier(fake_data)
        probs_e = netC(output_e)
        probs_e = F.softmax(probs_e, dim=1)

        entropies = calc_targeted_entropy(probs_e, true_label_g, targets, params['dis_c_dim'], device)
        loss_e = -torch.sum(entropies) #trying to maximize entropies
        loss_e = loss_e*beta*e_loose
        #Calculate Gradients
        #loss_e.backward()


        # Fake data treated as real.
        fake_data = netG(noise)
        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)

        isnan1 = torch.sum(torch.isnan(probs_fake))
        isnan2 = torch.sum(torch.isnan(label))
        if ((isnan1 > 0) or (isnan2 > 0)):
            print ('NAN VALUE in Generator Real Loss')

        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
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

        #Net loss for classifier
        C_loss = loss_c
        #Loss for Split
        #S_loss = loss_split
        S_loss = loss_e
        # Net loss for generator.
        gen_loss = gen_loss*d_loose
        G_loss = gen_loss + dis_loss + con_loss
        #G_loss = gen_loss + dis_loss + con_loss + align_loss
        #G_loss = dis_loss + con_loss
        G_loss = G_loss*alpha
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        nn.utils.clip_grad_value_(netG.parameters(), clip_value_2)
        nn.utils.clip_grad_value_(netQ.parameters(), clip_value_2)
        optimG.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tLoss_S: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item(), C_loss.item(), S_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())
        C_losses.append(C_loss.item())
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
            'discriminator' : discriminator.state_dict(),
            'classifier' : classifier.state_dict(),
            'netC' : netC.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
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
    'discriminator' : discriminator.state_dict(),
    'classifier' : classifier.state_dict(),
    'netC' : netC.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'params' : params
    }, 'checkpoint/model_final_fringe_{}'.format(params['dataset']))


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