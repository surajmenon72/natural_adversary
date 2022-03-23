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
import torch

import resnet

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
        default="True",
        type=str,
        choices=("True", "False"),
        help="should reset the k-means")

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


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

    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    assert missing_keys == [] and unexpected_keys == []

    batch_size = args.batch_size
    embedding_size = 1000
    head = nn.Linear(embedding, embedding_size)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    #model.cuda(gpu)
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
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.Grayscale(3), #hack to fit resnet to mnist
        transforms.ToTensor()])

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
    model.eval()
    num_classes = 10
    k_means = torch.zeros((num_classes, embedding_size))
    if (args.train_k_means == "True"):
        batches_to_avg = 100
        totals = torch.zeros((num_classes, 1))
        for i, (images, target) in enumerate(train_loader):
            print ('Batch')
            print (i)
            output = model(images.to(device))
            for j in range(batch_size):
                k_means[target[j], :] += output[j, :]
                totals[target[j], :] += 1

            if (i == batches_to_avg):
                break

        k_means = k_means/totals
        state = dict(
            k_means=k_means,
        )
        torch.save(state, args.exp_dir / "k_means.pth")
    else:
        k_dict = torch.load(args.exp_dir / "k_means.pth")
        k_means = k_dict["k_means"]
        print ('K_means loaded')

    print (k_means.shape)

    def calculate_fuzzy_k_means(model_output, k_means):
        b_size = model_output.shape[0]
        e_size = model_output.shape[1]
        k_size = k_means.shape[0]

        model_output_r = model_output.view((b_size, 1, e_size))
        k_means_r = k_means.view((1, k_size, e_size))
        k_means_r = k_means_r.repeat(b_size, 1, 1)

        distances = torch.cdist(model_output_r, k_means_r, p=2)
        distances = distances.view((b_size, k_size))

        sm_probs = torch.nn.functional.softmax(distances, dim=1)

        return sm_probs

    #Now see how well the K-means work
    total_correct = 0
    total_samples = 0
    batches_to_test = 10
    for i, (images, target) in enumerate(val_loader):
        print ('Val Batch')
        print (i)
        output = model(images.to(device))
        fuzzy_guesses = calculate_fuzzy_k_means(output, k_means)

        guesses = torch.argmax(fuzzy_guesses, dim=1)
        
        correct = (guesses == target)
        num_correct = torch.sum(correct, dim=0)

        total_correct += num_correct
        total_samples += batch_size

        if (i == batches_to_test):
            break

    accuracy = total_correct/total_samples

    print ('Validation Accuracy')
    print (accuracy)

if __name__ == "__main__":
    main()
