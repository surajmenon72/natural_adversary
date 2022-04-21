import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper, +FringeGAN experiments.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        #self.tconv1 = nn.ConvTranspose2d(82, 1024, 1, 1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(74, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        #img = torch.tanh(self.tconv4(x))
        img = torch.sigmoid(self.tconv4(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        # x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        # x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        #Remove batch norm
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.1, inplace=True)

        return x

    def get_feature_maps(self, x):
        fm = []
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        fm.append(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        fm.append(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        fm.append(x)

        return fm

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        #output = torch.sigmoid(self.conv(x))
        output = self.conv(x)
        output = torch.flatten(output)

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))

        return x

class CHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)

        return x

class Stretcher(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x, fm):
        # x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)  + fm[0]
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True) + fm[1]
        # x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True) + fm[2]

        #Remove batch norm
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True) + fm[0]
        x = F.leaky_relu(self.conv2(x), 0.1, inplace=True) + fm[1]
        x = F.leaky_relu(self.conv3(x), 0.1, inplace=True) + fm[2]

        return x

class HHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = self.conv(x)
        output = torch.flatten(output)

        return output
