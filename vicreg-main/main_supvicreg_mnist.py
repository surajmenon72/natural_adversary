# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms

import augmentations as aug
from distributed import init_distributed_mode

import resnet

import numpy as np
import matplotlib.pyplot as plt

from augmentations import SupConLoss


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=False,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="2048-2048-2048",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--encoder", default="base",
                        help='Type of encoder we should use')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=128,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser

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

def main(args):
    # torch.backends.cudnn.benchmark = True
    # init_distributed_mode(args)
    # print(args)
    # gpu = torch.device(args.device)

    # if args.rank == 0:
    #     args.exp_dir.mkdir(parents=True, exist_ok=True)
    #     stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    #     print(" ".join(sys.argv))
    #     print(" ".join(sys.argv), file=stats_file)

    if (args.encoder == "base"):
        transforms = aug.TrainTransformMNIST()
    elif (args.encoder == "resnet"):
        transforms = aug.TrainTransformMNISTResnet()
    else:
        print ("No valid Encoder given")
        exit()

    # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)

    args.rank = 0 #hack for now

    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print ('Using device')
    print (device)

    # transform = transforms.Compose([
    #     transforms.Resize(28),
    #     transforms.CenterCrop(28),
    #     transforms.ToTensor()])

    root = 'data/'
    # dataset = datasets.MNIST(root+'mnist/', train='train', 
    #                         download=True, transform=transforms)

    dataset = datasets.FashionMNIST(root+'fashionmnist/', train='train', 
                            download=True, transform=transforms)

    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = SupVICReg(args).to(device)

    sup_criterion = SupConLoss(temperature=0.07)

    #to extract backbone if necessary
    # ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    # torch.save(model.backbone.state_dict(), args.exp_dir / "resnet50.pth")
    # exit()

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # optimizer = LARS(
    #     model.parameters(),
    #     lr=0,
    #     weight_decay=args.wd,
    #     weight_decay_filter=exclude_bias_and_norm,
    #     lars_adaptation_filter=exclude_bias_and_norm,
    # )

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    #start_time = last_logging = time.time()
    start_time = time.time()
    #scaler = torch.cuda.amp.GradScaler()
    #alpha = 0.5
    alpha = 0.1
    for epoch in range(start_epoch, args.epochs):
        #sampler.set_epoch(epoch)
        print ('Epoch')
        print (epoch)
        epoch_start_time = time.time()
        #scheduler.step()
        for step, ((x, y), labels) in enumerate(loader):
            # x = x.cuda(gpu, non_blocking=True)
            # y = y.cuda(gpu, non_blocking=True)

            x = x.to(device)
            y = y.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            xy = torch.cat([x, y], dim=0)

            #lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            #     loss = model.forward(x, y)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            #loss = model.forward(x, y)


           #do VICREG
            x_v, y_v = model.forward_dual(x, y)

            vicreg_loss = model.loss_only(x_v, y_v)
            #vicreg_loss = torch.zeros(1)

            #do Supcon
            xy_s = torch.cat([x_v, y_v], dim=0)
            xy_s = F.normalize(xy_s, dim=1)
            x_s, y_s = torch.split(xy_s, [bsz, bsz], dim=0)

            xy_features_s = torch.cat([x_s.unsqueeze(1), y_s.unsqueeze(1)], dim=1)
            supcon_loss = sup_criterion(xy_features_s, labels)
            #supcon_loss = torch.zeros(1)

            if ((step % 50) == 0):
                print ('Current Vicreg Loss')
                print (vicreg_loss)
                print ('Current Supcon Loss')
                print (supcon_loss)

            loss = alpha*vicreg_loss + (1-alpha)*supcon_loss

            loss.backward()
            optimizer.step()

            # if args.rank == 0 and current_time - last_logging > args.log_freq_time:
            #     stats = dict(
            #         epoch=epoch,
            #         step=step,
            #         loss=loss.item(),
            #         time=int(current_time - start_time),
            #         lr=args.base_lr,
            #     )
            #     print(json.dumps(stats))
            #     print(json.dumps(stats), file=stats_file)
            #     last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
            torch.save(model.backbone.state_dict(), args.exp_dir / "model_backbone.pth")

        current_time = time.time()
        epoch_time = current_time - epoch_start_time
        print ("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    if args.rank == 0:
        torch.save(model.backbone.state_dict(), args.exp_dir / "model_backbone.pth")
        #torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


class SupVICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        # self.backbone, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        if (args.encoder == "base"):
            print ('Using Base Encoder')
            self.backbone = Encoder()
            self.embedding = 256
        elif (args.encoder == "resnet"):
            print ('Using Resnet Encoder')
            self.backbone = ResnetEncoder()
            self.embedding = 512
        else:
            print ("No Valid Encoder given")
            exit()
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss

    def forward_only(self, x):
        x = self.projector(self.backbone(x))

        return x

    def forward_dual(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        return x, y

    def loss_only(self, x, y):
        repr_loss = F.mse_loss(x, y)

        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
