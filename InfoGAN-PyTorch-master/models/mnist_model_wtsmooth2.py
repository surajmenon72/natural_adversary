import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

"""
Architecture based on InfoGAN paper, +FringeGAN experiments.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        #self.tconv1 = nn.ConvTranspose2d(82, 1024, 1, 1, bias=False)
        #self.tconv1 = nn.ConvTranspose2d(73, 1024, 1, 1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(256, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        #1 or 3 channels
        #self.tconv4 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        #img = torch.tanh(self.tconv4(x))
        img = torch.sigmoid(self.tconv4(x))

        return img

    def f_logits(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        logits = self.tconv4(x)

        return logits

class Generator_Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(512, 448, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(448)

        self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        img = torch.tanh(self.tconv5(x))

    def f_logits(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        logits = self.tconv5(x)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        #1 or 3 channels
        #self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        #Remove batch norm
        # x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        # x = F.leaky_relu(self.conv2(x), 0.1, inplace=True)
        # x = F.leaky_relu(self.conv3(x), 0.1, inplace=True)

        return x

    def get_feature_maps(self, x):
        fm = []
        x = self.conv1(x)
        fm.append(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.conv2(x)
        fm.append(x)
        x = F.leaky_relu(self.bn2(x), 0.1, inplace=True)
        x = self.conv3(x)
        fm.append(x)
        x = F.leaky_relu(self.bn3(x), 0.1, inplace=True)

        # fm = []
        # x = self.conv1(x)
        # fm.append(x)
        # x = F.leaky_relu(x, 0.1, inplace=True)
        # x = self.conv2(x)
        # fm.append(x)
        # x = F.leaky_relu(x, 0.1, inplace=True)
        # x = self.conv3(x)
        # fm.append(x)
        # x = F.leaky_relu(x, 0.1, inplace=True)

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

class DHead_KL(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

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

        #1 or 3 channels
        #self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
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


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        model="resnet18",
        use_pretrained=False,
        **kwargs,
    ):
        super().__init__()

        encoder = eval(model)(pretrained=use_pretrained)
        self.f = []
        """for name, module in encoder.named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.f.append(module)"""
        for name, module in encoder.named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        self.feature_size = encoder.fc.in_features
        self.d_model = encoder.fc.in_features

    def forward(self, x):
        x = self.f(x)
        x = torch.flatten(x, start_dim=1)
        return x

class CHead(nn.Module):
    def __init__(self):
        super().__init__()

        #self.fc1 = nn.Linear(512, 10)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)

        return x

class Stretcher(nn.Module):
    def __init__(self):
        super().__init__()

        #1 or 3 channels
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        #self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x, fm):
        x = F.leaky_relu(self.bn1(self.conv1(x) + fm[0]), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x) + fm[1]), 0.1, inplace=True) 
        x = F.leaky_relu(self.bn3(self.conv3(x) + fm[2]), 0.1, inplace=True) 

        #Remove batch norm
        # x = F.leaky_relu(self.conv1(x) + fm[0], 0.1, inplace=True)
        # x = F.leaky_relu(self.conv2(x) + fm[1], 0.1, inplace=True)
        # x = F.leaky_relu(self.conv3(x) + fm[2], 0.1, inplace=True)

        return x

class HHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = self.conv(x)
        output = torch.flatten(output)

        return output
