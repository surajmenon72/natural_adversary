import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper, +FringeGAN experiments.
"""

# class Generator(nn.Module):
#     r""" An Generator model.
#     `Generative Adversarial Networks model architecture from the One weird trick...
#     <https://arxiv.org/abs/1704.00028v3>`_ paper.
#     """

#     def __init__(self):
#         super(Generator, self).__init__()

#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(73, 512, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(64, 1, 4, 2, 1),
#             nn.Tanh()
#         )

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         r"""Defines the computation performed at every call.
#         Args:
#             input (tensor): input tensor into the calculation.
#         Returns:
#             A four-dimensional vector (NCHW).
#         """
#         out = self.main(input)
#         return out

# class Discriminator(nn.Module):
#     """ An Discriminator model.
#     `Generative Adversarial Networks model architecture from the One weird trick...
#     <https://arxiv.org/abs/1704.00028v3>`_ paper.
#     """

#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.main = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, True),

#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, True),

#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, True),

#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, True),

#             #nn.Conv2d(512, 1, 4, 1, 0),
#         )

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """ Defines the computation performed at every call.
#         Args:
#             input (tensor): input tensor into the calculation.
#         Returns:
#             A four-dimensional vector (NCHW).
#         """
#         out = self.main(input)
#         #out = torch.flatten(out)
#         return out

# class DHead(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Conv2d(512, 1, 4, 1, 0)

#     def forward(self, x):
#         out = self.conv(x)
#         out = torch.flatten(out)

#         return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        #self.tconv1 = nn.ConvTranspose2d(82, 1024, 1, 1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(73, 1024, 1, 1, bias=False)
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

        img = torch.tanh(self.tconv4(x))

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
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        #output = torch.sigmoid(self.conv(x))
        output = self.conv(x)
        output = torch.flatten(output)

        return output

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

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 10, 1)
        self.conv_var = nn.Conv2d(128, 10, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
