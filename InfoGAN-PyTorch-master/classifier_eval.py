from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
import copy

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
        help="should reset the k-means"
    )
    parser.add_argument(
        "--use_base_resnet",
        default="resnet",
        type=str,
        choices=("base", "resnet"),
        help="use base or resnet model"
    )
    parser.add_argument(
        "--use_thanos_vicreg",
        default="vicreg",
        type=str,
        choices=("thanos", "vicreg"),
        help="load thanos or vicreg encoder"
    )
    parser.add_argument(
        "--train_knn",
        default='False',
        type=str,
        choices=('False', 'True'),
        help="should train the knn"
    )
    parser.add_argument(
        "--validate_knn",
        default='False',
        type=str,
        choices=('False', 'True'),
        help="should train the knn"
    )
    parser.add_argument(
        "--load_model",
        default='False',
        type=str,
        choices=('False', 'True'),
        help="should load or use random encoder"
    )

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

def calculate_fuzzy_knn(model_output, knn_e, knn_t, k=100, num_classes=10):
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

def calculate_fuzzy_knn_eff(model_output, knn_e, knn_t, k=100, num_classes=10):
    b_size = model_output.shape[0]
    e_size = model_output.shape[1]
    knn_size = knn_e.shape[0]

    distances = torch.zeros((b_size, knn_size))
    knn_e_r = knn_e.view((1, knn_size, e_size))
    for b in range(b_size):
        output_chunk = model_output[b, :]
        output_chunk_r = output_chunk.view((1, 1, e_size))
        distance_b = torch.cdist(output_chunk_r, knn_e_r, p=2)
        distance_b = distance_b.view((knn_size))
        distances[b, :] = copy.deepcopy(distance_b[:])

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

def calc_entropy(dist):
    safe_dist = F.normalize(dist+1e-9)
    print (dist[0])
    print (safe_dist[0])
    log_dist = torch.log(safe_dist)
    mult = dist*log_dist
    entropy = -torch.sum(mult, dim=1)
    return entropy


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

    # use_base_resnet = 'resnet'
    # use_thanos_vicreg = 'vicreg'
    # load_model = True
    # train_knn = False

    exp_dir = './checkpoints/knn.pth'
    use_base_resnet = args.use_base_resnet
    use_thanos_vicreg = args.use_thanos_vicreg

    load_model = False
    if (args.load_model == 'True'):
        load_model = True

    train_knn = False
    if (args.train_knn == 'True'):
        train_knn = True

    validate_knn = False
    if (args.validate_knn == 'True'):
        validate_knn = True

    backbone = None
    if (use_base_resnet == 'resnet'):
        backbone = ResnetEncoder()
    else:
        backbone = Encoder()

    if (load_model == True):
        if (use_thanos_vicreg == 'thanos'):
            if (use_base_resnet == 'resnet'):
                ckpt = './checkpoints/thanos_resnet_15.ckpt'
                loaded = torch.load(ckpt, map_location=torch.device('cpu'))
                backbone.load_state_dict(
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
            else:
                ckpt = './checkpoints/thanos_base_fashion_30.ckpt'
                loaded = torch.load(ckpt, map_location=torch.device('cpu'))
                backbone.load_state_dict(
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
        else:
            if (use_base_resnet == 'resnet'):
                ckpt = './checkpoints/vicreg_backbone_resnet_60.pth'
                loaded = torch.load(ckpt, map_location=torch.device('cpu'))
                missing_keys, unexpected_keys = backbone.load_state_dict(loaded, strict=False)
            else:
                #ckpt = './checkpoints/vicreg_backbone_base_60.pth'
                #ckpt = './checkpoints/vicreg_backbone_base_fashion_60.pth'
                ckpt = './checkpoints/supvic_backbone_base_fashion_.85_157.pth'
                loaded = torch.load(ckpt, map_location=torch.device('cpu'))
                missing_keys, unexpected_keys = backbone.load_state_dict(loaded, strict=False)

        print ('Model Loaded!')

    batch_size = 128
    if (use_base_resnet == 'resnet'):
        embedding_size = 512
    else:
        embedding_size = 256
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


    #Load MNIST
    transform = None
    if (use_base_resnet == 'resnet'): 
        transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]
                ),]
        )
    else:
        transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),]
        )

    root = 'data/'
    # train_dataset = datasets.MNIST(root+'mnist/', train=True, 
    #                         download=True, transform=transform)
    # val_dataset = datasets.MNIST(root+'mnist/', train=False, 
    #                         download=True, transform=transform)
    train_dataset = datasets.FashionMNIST(root+'fashionmnist/', train=True, 
                            download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root+'fashionmnist/', train=False, 
                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)


    #Set knn,works only if targets are about evenly distributed in training set
    model.eval()
    batches_for_knn = len(train_loader)-1
    knn_e = torch.zeros((batches_for_knn*batch_size, embedding_size))
    knn_t = torch.zeros(batches_for_knn*batch_size)

    if (train_knn == True):
        print ('Training KNN')
        for i, (images, target) in enumerate(train_loader):
            print ('Batch')
            print (i)
            output = model(images.to(device))

            start_index = i*batch_size
            end_index = (i+1)*batch_size

            knn_e[start_index:end_index, :] = copy.deepcopy(output[:, :])
            knn_t[start_index:end_index] = copy.deepcopy(target[:])

            if (i == (batches_for_knn-1)):
                break

        state = dict(
            knn_e = knn_e,
            knn_t = knn_t,
        )
        torch.save(state, exp_dir)
    else:
        knn_dict = torch.load(exp_dir)
        knn_e = copy.deepcopy(knn_dict["knn_e"])
        knn_t = copy.deepcopy(knn_dict["knn_t"])
        print ('KNN loaded')

    # image = torch.load('3-9.pt')
    # image = torch.cat([image, image, image], dim=0) 
    # image.resize_(1, 3, 28, 28)
    # img_tensor = image.float()

    #Now lets validate w/ KNN
    if (validate_knn == True):
        batches_to_test = 10
        total_correct = 0
        total_samples = 0
        total_entropy = 0
        print ('Validating w/ KNN')
        targets = torch.zeros((10))
        for i, (images, target) in enumerate(val_loader):
            print ('Val Batch')
            print (i)

            output = model(images.to(device))

            # img_test = img_tensor
            # output = model(img_test.to(device))
            #fuzzy_guesses = calculate_fuzzy_knn(output, knn_e, knn_t, k=100)
            fuzzy_guesses = calculate_fuzzy_knn_eff(output, knn_e, knn_t, k=100)

            # print (fuzzy_guesses.shape)
            # print (fuzzy_guesses[0])
            # exit()

            guesses = torch.argmax(fuzzy_guesses, dim=1)
            
            correct = (guesses == target)
            num_correct = torch.sum(correct, dim=0)

            total_correct += num_correct
            total_samples += batch_size

            print (fuzzy_guesses.shape)
            print (fuzzy_guesses[0])
            entropy = calc_entropy(fuzzy_guesses)
            print (entropy[0])
            print (torch.sum(entropy))
            exit()
            total_entropy += entropy


            for b in range(batch_size):
                index = int(target[b])
                targets[index] += 1

            if (i == batches_to_test):
                break

        accuracy = total_correct/total_samples
        avg_entropy = total_entropy/total_samples

        print ('Validation Accuracy w/ KNN')
        print (accuracy)

        print ('Target Distribution')
        print (targets)

        print ('Avg Entropy')
        print (avg_entropy)


if __name__ == "__main__":
    main()
