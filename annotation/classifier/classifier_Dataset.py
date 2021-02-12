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


class S3d_Normal_Dataset(torch.utils.data.Dataset):
    def __init__(self, coco_json_path, transform):

        with open(coco_json_path, 'rb') as j:
            coco_info = json.loads(j.read())
        self.img_normalClass_list = []
        for item in coco_info["annotations"]:  
            self.img_normalClass_list.append([item["image_path"], item["anchor_normal_class"]])
        self.transform = transform

    def __getitem__(self, index):

        img_path , anchor_normal_class = self.img_normalClass_list[index]
        img = Image.open(img_path).convert('RGB') #plt.imread(img_path) #.crop((left, top, right, bottom))
        img = self.transform(img)
        # print(img.shape) #chirs: debug
        # print("testing : ", anchor_normal_class)
        return (img_path, img , anchor_normal_class) # TODO

    def __len__(self):
        return len(self.img_normalClass_list)


class Pos_Neg_eql_Dataset(torch.utils.data.Dataset):
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
        self.neg_index_to_read = random.sample(range(0,len(self.neg_img_list)),len(self.neg_img_list))
        self.transform = transform

    def __getitem__(self, index):
        if index % 2 == 0: # get positive sample (label : 1)
            img_path = self.pos_img_list[int(index/2)]
            random.seed()
            
            img = Image.open(img_path).convert('RGB') 
            img = self.transform(img)
            return (img_path, img , 1) 
        else: # get negative sample (label : 0)
            if len(self.neg_index_to_read) == 0:
                self.neg_index_to_read = random.sample(range(0,len(self.neg_img_list)),len(self.neg_img_list))
            img_id = self.neg_index_to_read.pop()
            img_path = self.neg_img_list[img_id]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return (img_path,img, 0) 
    

    def __len__(self):
        return 2*len(self.pos_img_list)


class S3d_Cls_Dataset_neg(torch.utils.data.Dataset):
    def __init__(self, neg_list_path, transform):

        with open(neg_list_path) as file:
            lines = file.readlines()
        self.neg_img_list = [line.strip() for line in lines]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.neg_img_list[index])
        img = self.transform(img)
        return (self.neg_img_list[index], img , 0) # TODO


    def __len__(self):
        return len(self.neg_img_list)


class Dataset_to_label(torch.utils.data.Dataset):
    def __init__(self, unlabeled_list_path, transform):

        with open(unlabeled_list_path) as file:
            lines = file.readlines()
        self.unlabeled_img_list = [line.strip() for line in lines]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.unlabeled_img_list[index])
        img = self.transform(img)
        return (self.unlabeled_img_list[index], img , 0) # TODO


    def __len__(self):
        return len(self.unlabeled_img_list)

class S3d_Cls_Dataset_noRandom(torch.utils.data.Dataset):
    def __init__(self, pos_list_path, neg_list_path, transform):
        self.img_list = []
        # loading positive
        if os.path.isfile(pos_list_path):
            with open(pos_list_path) as file:
                lines = file.readlines()
            self.img_list += [[line.strip(),1] for line in lines] 


        # loading negative
        if os.path.isfile(neg_list_path):
            with open(neg_list_path) as file:
                lines = file.readlines()
            self.img_list += [[line.strip(),0] for line in lines] 
        

        self.transform = transform

    def __getitem__(self, index):
        img_path , label = self.img_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return (img_path, img ,label) # TODO

        
    def __len__(self):
        return len(self.img_list)



class S3d_Cls_Dataset_diff_len(torch.utils.data.Dataset):
    def __init__(self, pos_list_path, neg_list_path, transform,data_len):
        self.img_list = []
        # loading positive
        random.seed(5)
        if os.path.isfile(pos_list_path):
            with open(pos_list_path) as file:
                lines = file.readlines()
            index_to_read = random.sample(range(0,len(lines)),data_len)
            self.img_list += [[lines[i].strip(),1] for i in index_to_read] 

        # print(index_to_read)

        # loading negative
        random.seed(5)
        if os.path.isfile(neg_list_path):
            with open(neg_list_path) as file:
                lines = file.readlines()
            index_to_read = random.sample(range(0,len(lines)),data_len)
            self.img_list += [[lines[i].strip(),0] for i in index_to_read] 
        # print(index_to_read)

        self.transform = transform

    def __getitem__(self, index):
        img_path , label = self.img_list[index]
        
        random.seed()
        left = random.randint(0, 128)
        top = random.randint(0, 96)
        right = random.randint(512, 640)
        bottom = random.randint(384, 480)
        print("image pos : ",left, top, right, bottom)
        img = Image.open(img_path).crop((left, top, right, bottom))
        img = self.transform(img)
   
        return (img_path, img ,label) # TODO

        
    def __len__(self):
        return len(self.img_list)
