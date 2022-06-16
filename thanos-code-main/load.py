import torch
import hydra
import pytorch_lightning as pl

ckpt = './models/adv_pths_best-resnet18.ckpt'

loaded = torch.load(ckpt, map_location=torch.device('cpu'))

print (loaded)