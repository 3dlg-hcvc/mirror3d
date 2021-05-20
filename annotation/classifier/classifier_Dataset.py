import argparse
import os
import random
import shutil
import time
import warnings
import matplotlib.pyplot as plt

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import random
import json
from skimage import io


class PosNegEqlDataset(torch.utils.data.Dataset):
    def __init__(self, pos_list_path, neg_list_path, transform):

        with open(pos_list_path) as file:
            lines = file.readlines()
        self.pos_img_list = []
        for line in lines:
            if os.path.exists(line.strip()):
                self.pos_img_list.append(line.strip())
        # self.pos_img_list = [line.strip() for line in lines]

        with open(neg_list_path) as file:
            lines = file.readlines()
        self.neg_img_list = []
        for line in lines:
            if os.path.exists(line.strip()):
                self.neg_img_list.append(line.strip())
        # self.neg_img_list = [line.strip() for line in lines]
        self.neg_index_to_read = random.sample(range(0, len(self.neg_img_list)), len(self.neg_img_list))
        self.transform = transform

    def __getitem__(self, index):
        if index % 2 == 0:  # get positive sample (label : 1)
            img_path = self.pos_img_list[int(index / 2)]
            random.seed()

            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img_path, img, 1
        else:  # get negative sample (label : 0)
            if len(self.neg_index_to_read) == 0:
                self.neg_index_to_read = random.sample(range(0, len(self.neg_img_list)), len(self.neg_img_list))
            img_id = self.neg_index_to_read.pop()
            img_path = self.neg_img_list[img_id]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img_path, img, 0

    def __len__(self):
        return 2 * len(self.pos_img_list)


class DatasetToLabel(torch.utils.data.Dataset):
    def __init__(self, unlabeled_list_path, transform):
        with open(unlabeled_list_path) as file:
            lines = file.readlines()
        self.unlabeled_img_list = [line.strip() for line in lines]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.unlabeled_img_list[index])
        img = self.transform(img)
        return self.unlabeled_img_list[index], img, 0

    def __len__(self):
        return len(self.unlabeled_img_list)
