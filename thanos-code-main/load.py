import torch
import hydra
import pytorch_lightning as pl

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50  # noqa: F401

import torchvision.transforms as transforms
import torchvision.datasets as dsets

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

class VisionEncoder(nn.Module):
	def __init__(self):
		super().__init__()

		model = ResnetEncoder()

	def forward(self, x):
		result = model.forward(x)
		return result

ckpt = './models/adv_pths_best-resnet18.ckpt'
loaded = torch.load(ckpt, map_location=torch.device('cpu'))

# print (loaded.keys())
# print (loaded['state_dict'].keys())

#e = VisionEncoder()
e = ResnetEncoder()
#e.load_state_dict(loaded['state_dict'])
e.load_state_dict(
    {
        ".".join(k.split(".")[3:]): v
        for k, v in loaded["state_dict"].items()
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

print ('Loaded State Dict')
exit()

#Now test running an image through

root = 'data/'
transform = transforms.Compose([
        	transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])
dataset = dsets.MNIST(root+'mnist/', train='train', 
                    download=True, transform=transform)




