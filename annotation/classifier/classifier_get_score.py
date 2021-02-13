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
from mirror3d_resnet import resnet50
from PIL import Image
import random
from tensorboardX import SummaryWriter
import numpy as np
from classifier_Dataset import *
import operator

import datetime as d
from tqdm import tqdm
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
import json


def eval_get_score(args, val_loader, criterion, model):
    # TODO append image_list and score to txt (sort the score as well)
    img_score = dict()
    for i, (img_path_list, images, _) in enumerate(tqdm(val_loader)):
        
        output = model(images)
        output = torch.nn.functional.softmax(output,dim=1)

        obj_score = output[:,-1]
        for index, img_path in enumerate(img_path_list):
            img_score[img_path] = obj_score[index].item()

    img_score = dict(sorted(img_score.items(), key=operator.itemgetter(1),reverse=True))
    json_save_path = os.path.join(args.output_save_folder, "imgPath_score_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  + ".json")
    save_json(json_save_path,img_score)


def save_json(save_path,data):
    out_json = json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    with open(save_path, "w") as fo:
        fo.write(out_json)
        fo.close()
        print("json file saved to : ",save_path )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # # parser.add_argument('--num_classes', default=2, type=int, help='num_classes to classify')
    parser.add_argument('--batch_size', default=16, type=int)
    # TODO retrain
    parser.add_argument('--img_list_to_test',  default="/local-scratch/share_data/scannet_extracted/info/scannet_frames_25k.txt", type=str)
    # TODO retrain
    parser.add_argument('--resume_path',  default="/project/3dlg-hcvc/jiaqit/output/classifier/checkpoint/epoch_20_checkpoint.pth.tar", type=str) # /local-scratch/jiaqit/exp/chris_planercnn/examples/log/retrain_7_2020_08_07-16_45_00_log/checkpoint.pth.tar
    # TODO retrain
    parser.add_argument('--output_save_folder', default="/project/3dlg-hcvc/jiaqit/output/classifier/checkpoint/scannet_list_sort", type=str)

    args = parser.parse_args(args=[])

    os.makedirs(args.output_save_folder, exist_ok=True)
    print(args)


    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    # TODO change num_classes
    model = resnet50()
    model.eval()
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
    # used for visulize image
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )


    train_sampler = None
    val_dataset = Dataset_to_label(args.img_list_to_test,transform) 
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=1, pin_memory=True, sampler=train_sampler)


    model = torch.nn.DataParallel(model).cuda()


    if args.resume_path:
        print("=> loading checkpoint '{}'".format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_path, checkpoint['epoch']))
    else:
        print("------- wihout pretrained checkpoint")
    cudnn.benchmark = True

    eval_get_score(args, val_loader, criterion, model)