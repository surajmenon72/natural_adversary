import os

import numpy as np


def sparse2coarse(targets, scramble=False):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    sparse_coarse_array = [
        14,
        31,
        16,
        16,
        8,
        34,
        35,
        0,
        0,
        0,
        4,
        3,
        22,
        14,
        7,
        23,
        23,
        23,
        9,
        9,
        3,
        3,
        3,
        21,
        11,
        11,
        11,
        11,
        11,
        11,
        6,
        6,
        6,
        39,
        39,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        12,
        30,
        19,
        37,
        37,
        37,
        37,
        37,
        28,
        28,
        28,
        37,
        2,
        61,
        46,
        53,
        46,
        41,
        53,
        45,
        45,
        47,
        43,
        53,
        65,
        45,
        56,
        52,
        46,
        62,
        58,
        41,
        49,
        62,
        62,
        64,
        45,
        62,
        66,
        46,
        61,
        50,
        62,
        47,
        49,
        45,
        50,
        45,
        65,
        65,
        58,
        53,
        53,
        55,
        62,
        58,
        58,
        64,
        48,
        46,
        41,
        65,
        44,
        61,
        50,
        65,
        46,
        49,
        65,
        44,
        65,
        62,
        58,
        46,
        46,
        65,
        62,
        41,
        45,
        55,
        55,
        50,
        50,
        51,
        47,
        62,
        60,
        65,
        46,
        52,
        62,
        66,
        60,
        61,
        53,
        50,
        53,
        43,
        46,
        65,
        60,
        61,
        60,
        46,
        54,
        50,
        58,
        65,
        58,
        50,
        46,
        58,
        46,
        62,
        48,
        63,
        45,
        62,
        65,
        58,
        65,
        61,
        41,
        46,
        58,
        43,
        47,
        58,
        48,
        48,
        59,
        48,
        52,
        52,
        52,
        52,
        52,
        38,
        38,
        18,
        17,
        17,
        17,
        17,
        52,
        52,
        52,
        52,
        58,
        58,
        7,
        58,
        58,
        27,
    ]

    targets = np.array(sparse_coarse_array)[targets]
    return targets.tolist()


def create_val_img_folder(path_to_dataset: str):
    """
    This method is responsible for separating validation images into
    separate sub folders
    """
    val_dir = os.path.join(path_to_dataset, "val")
    img_dir = os.path.join(val_dir, "images")

    fp = open(os.path.join(val_dir, "val_annotations.txt"), "r")
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = os.path.join(img_dir, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))
