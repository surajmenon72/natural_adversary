import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)

    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
        
        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx

def noise_sample_target(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device, targets):
    """
    Sample random noise vector for training. Assuming we are training for targets

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    targets : Which continuous varible we will use
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    c_nums = []

    #should be the same as the original infoGAN
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
        
        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)


    #should only make non-zero the target class var
    if (n_con_c != 0):
        con_c = torch.zeros((batch_size, n_con_c, 1, 1), device=device)
        #Random uniform between 0 and 1 IF the target is set for that batch
        for i in range(batch_size):
            t = targets[i]

            num = torch.rand(1, 1, 1, 1)
            con_c[i, t, :, :] = num
            c_nums.append(num)


    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx, c_nums


def get_targets(true_labels, num_classes, device):
    """
    Get target classes for which to adversarially train.

    Takes in the true_labels and total classes and then outputs a tensor of random targets
    """
    b_size = true_labels.shape[0]
    labels = nn.functional.one_hot(true_labels)
    targets = torch.randint(0, num_classes, (b_size, ), device=device)

    return labels, targets

def get_split_labels(true_label, targets, c_nums, num_classes, device):
    """
    Get split classification labels, weighted by the random number generated for continuous variable

    Returns the weighted labels, including weights only for the true label the random target
    """
    b_size = true_label.shape[0]
    labels = torch.zeros((b_size, num_classes), device=device)
    for i in range(b_size):
        c_num = c_nums[i]
        #c_num /= 2 #divide in half as original is between 0-1, we want between 0-0.5

        tl = true_label[i]
        t = targets[i]
        if (tl != t):
            labels[i, tl] = 1-c_num
            labels[i, t] = c_num
        else:
            labels[i, tl] = 1

    return labels

def calc_entropy(dist):
    log_dist = torch.log(dist)
    mult = dist*log_dist
    entropy = torch.sum(mult, dim=1)
    return entropy

