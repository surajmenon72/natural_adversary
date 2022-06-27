from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn.functional as F
import torch

# import resnet
# import main_vicreg_mnist 

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        metavar="B",
        help="batch training/val size"
        )
    parser.add_argument(
        "--train_k_means",
        default="False",
        type=str,
        choices=("True", "False"),
        help="should reset the k-means")
    parser.add_argument(
        "--train_knn",
        default="False",
        type=str,
        choices=("True", "False"),
        help="should train the knn"),

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser

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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 512)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))

        return x

def main():
    parser = get_arguments()
    args = parser.parse_args()
    # if args.train_percent in {1, 10}:
    #     args.train_files = urllib.request.urlopen(
    #         f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
    #     ).readlines()
    # args.ngpus_per_node = torch.cuda.device_count()
    # if "SLURM_JOB_ID" in os.environ:
    #     signal.signal(signal.SIGUSR1, handle_sigusr1)
    #     signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    # args.rank = 0
    # args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    # args.world_size = args.ngpus_per_node
    #torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    main_worker(args)




#def main_worker(gpu, args):
def main_worker(args):
    # args.rank += gpu
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     init_method=args.dist_url,
    #     world_size=args.world_size,
    #     rank=args.rank,
    # )

    # if args.rank == 0:
    #     args.exp_dir.mkdir(parents=True, exist_ok=True)
    #     stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    #     print(" ".join(sys.argv))
    #     print(" ".join(sys.argv), file=stats_file)

    # torch.cuda.set_device(gpu)
    # torch.backends.cudnn.benchmark = True

    #device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    device = torch.device('cpu')

    # backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    # backbone = ResnetEncoder()
    # embedding = 256 
    # state_dict = torch.load(args.pretrained, map_location="cpu")
    # missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    # assert missing_keys == [] and unexpected_keys == []

    ckpt = './models/adv_pths_best-resnet18.ckpt'
    loaded = torch.load(ckpt, map_location=torch.device('cpu'))

    backbone = ResnetEncoder()
    #backbone = Encoder()
    # backbone.load_state_dict(
    #     {
    #         ".".join(k.split(".")[3:]): v
    #         for k, v in loaded["state_dict"].items()
    #         if (
    #             # source_module in k
    #             # and "model" in k
    #             # and k.split(".")[2] == source_module
    #             "model" in k
    #             and "ImageEncoder" in k
    #         )
    #     },
    #     strict=True,
    # )

    batch_size = 16
    embedding_size = 512
    output_size = 10
    head = nn.Linear(embedding_size, output_size)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    #model = nn.Sequential(backbone, head)
    #model.cuda(gpu)
    model = backbone
    #embedding_size = embedding
    model.to(device)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss()

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    # if (args.exp_dir / "checkpoint.pth").is_file():
    #     ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
    #     start_epoch = ckpt["epoch"]
    #     best_acc = ckpt["best_acc"]
    #     k_means = ckpt["k_means"]
    #     model.load_state_dict(ckpt["model"])
    #     optimizer.load_state_dict(ckpt["optimizer"])
    #     scheduler.load_state_dict(ckpt["scheduler"])
    # else:
    #     start_epoch = 0
    #     best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    # traindir = args.data_dir / "train"
    # valdir = args.data_dir / "val"
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose(
    #         [
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     ),
    # )
    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     ),
    # )



    # if args.train_percent in {1, 10}:
    #     train_dataset.samples = []
    #     for fname in args.train_files:
    #         fname = fname.decode().strip()
    #         cls = fname.split("_")[0]
    #         train_dataset.samples.append(
    #             (traindir / cls / fname, train_dataset.class_to_idx[cls])
    #         )

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # kwargs = dict(
    #     batch_size=args.batch_size // args.world_size,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, sampler=train_sampler, **kwargs
    # )
    # val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    #Load MNIST
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]
            ),]
    )

    root = 'data/'
    train_dataset = datasets.MNIST(root+'mnist/', train='train', 
                            download=True, transform=transform)
    val_dataset = datasets.MNIST(root+'mnist/', train=False, 
                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    #Set the k-means
    # num_classes = 10
    # batches_to_avg = 100
    # k_means = torch.zeros((num_classes, embedding_size))
    # if (args.train_k_means == "True"):
    #     print ('Training K-Means')
    #     totals = torch.zeros((num_classes, 1))
    #     for i, (images, target) in enumerate(train_loader):
    #         print ('Batch')
    #         print (i)
    #         output = model(images.to(device))
    #         for j in range(batch_size):
    #             k_means[target[j], :] += output[j, :]
    #             totals[target[j], :] += 1

    #         if (i == (batches_to_avg-1)):
    #             break

    #     k_means = k_means/totals
    #     state = dict(
    #         k_means=k_means,
    #     )
    #     torch.save(state, args.exp_dir / "k_means.pth")
    # else:
    #     k_dict = torch.load(args.exp_dir / "k_means.pth")
    #     k_means = k_dict["k_means"]
    #     print ('K_means loaded')

    # print (k_means.shape)

    #Set knn,works only if targets are about evenly distributed in training set
    model.eval()
    batches_for_knn = 400
    exp_dir = './models/knn.pth'
    train_knn = 'True'
    knn_e = torch.zeros((batches_for_knn*batch_size, embedding_size))
    knn_t = torch.zeros(batches_for_knn*batch_size)

    if (train_knn == "True"):
        print ('Training KNN')
        for i, (images, target) in enumerate(train_loader):
            print ('Batch')
            print (i)
            output = model(images.to(device))

            start_index = i*batch_size
            end_index = (i+1)*batch_size

            knn_e[start_index:end_index, :] = output[:, :]
            knn_t[start_index:end_index] = target[:]

            if (i == (batches_for_knn-1)):
                break

        state = dict(
            knn_e = knn_e,
            knn_t = knn_t,
        )
        torch.save(state, exp_dir)
    else:
        knn_dict = torch.load(exp_dir)
        knn_e = knn_dict["knn_e"]
        knn_t = knn_dict["knn_t"]
        print ('KNN loaded')

    # def calculate_fuzzy_k_means(model_output, k_means):
    #     b_size = model_output.shape[0]
    #     e_size = model_output.shape[1]
    #     k_size = k_means.shape[0]

    #     model_output_r = model_output.view((b_size, 1, e_size))
    #     k_means_r = k_means.view((1, k_size, e_size))
    #     k_means_r = k_means_r.repeat(b_size, 1, 1)

    #     distances = torch.cdist(model_output_r, k_means_r, p=2) #verified this works
    #     distances = distances.view((b_size, k_size))

    #     sm_probs = torch.nn.functional.softmax(distances, dim=1)

    #     return sm_probs

    #Validation tasks

    # #Now see how well the K-means work
    # batches_to_test = 10
    # total_correct = 0
    # total_samples = 0
    # print ('Validating on K-Means')
    # for i, (images, target) in enumerate(val_loader):
    #     print ('Val Batch')
    #     print (i)
    #     output = model(images.to(device))
    #     fuzzy_guesses = calculate_fuzzy_k_means(output, k_means)

    #     guesses = torch.argmax(fuzzy_guesses, dim=1)
    #     print (guesses)
        
    #     correct = (guesses == target)
    #     num_correct = torch.sum(correct, dim=0)

    #     total_correct += num_correct
    #     total_samples += batch_size

    #     if (i == batches_to_test):
    #         break

    # accuracy = total_correct/total_samples

    # print ('Validation Accuracy w/ K-Means')
    # print (accuracy)

    def calculate_fuzzy_knn(model_output, knn_e, knn_t, k=5000, num_classes=10):
        b_size = model_output.shape[0]
        e_size = model_output.shape[1]
        knn_size = knn_e.shape[0]

        model_output_r = model_output.view((b_size, 1, e_size))
        knn_e_r = knn_e.view((1, knn_size, e_size))
        knn_e_r = knn_e_r.repeat(b_size, 1, 1)

        distances = torch.cdist(model_output_r, knn_e_r, p=2) #verified this works
        distances = distances.view((b_size, knn_size))

        knn_guesses = torch.zeros((b_size, num_classes))
        for i in range(b_size):
            d = distances[i, :]
            d_s, d_s_i = torch.sort(d)
            max_val = torch.max(d_s)
            for j in range(k):
                knn_guesses[i, int(knn_t[d_s_i[j]])] += max_val - d_s[j]

            #knn_guesses[i, :] /= torch.max(knn_guesses[i, :])

        #print (knn_guesses[0])

        #sm_knn = torch.nn.functional.softmax(knn_guesses, dim=1)

        #sm_knn = knn_guesses/torch.sum(knn_guesses, dim=1)
        sm_knn = knn_guesses/(torch.sum(knn_guesses, dim=1).view((knn_guesses.shape[0], 1)))

        return sm_knn

    # image = torch.load('3-9.pt')
    # image = torch.cat([image, image, image], dim=0) 
    # image.resize_(1, 3, 28, 28)
    # img_tensor = image.float()

    #Now lets validate w/ KNN
    batches_to_test = 100
    total_correct = 0
    total_samples = 0
    print ('Validating w/ KNN')
    for i, (images, target) in enumerate(val_loader):
        print ('Val Batch')
        print (i)

        output = model(images.to(device))

        # img_test = img_tensor
        # output = model(img_test.to(device))
        fuzzy_guesses = calculate_fuzzy_knn(output, knn_e, knn_t)

        # print (fuzzy_guesses.shape)
        # print (fuzzy_guesses[0])
        # exit()

        guesses = torch.argmax(fuzzy_guesses, dim=1)
        
        correct = (guesses == target)
        num_correct = torch.sum(correct, dim=0)

        total_correct += num_correct
        total_samples += batch_size

        if (i == batches_to_test):
            break

    accuracy = total_correct/total_samples

    print ('Validation Accuracy w/ KNN')
    print (accuracy)

if __name__ == "__main__":
    main()
