# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.Mirror3D_eval import Mirror3d_eval

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *
sys.path.append('./')
from important_model.bts_nyu_v2_pytorch_densenet161.bts_nyu_v2_pytorch_densenet161 import BtsModel

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.resume_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    if args.coco_input:
        train_info_to_eval = []
        for one_cal_json in args.coco_val.split(","):
            one_cal_json = one_cal_json.strip()
            one_images = read_json(one_cal_json)["images"]
            train_info_to_eval += one_images
        num_test_samples = len(train_info_to_eval)
    else:
        num_test_samples = get_num_lines(args.filenames_file)

    lines = ["None None"] * num_test_samples
    if not args.coco_input :
        with open(args.filenames_file) as f:
            lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.resume_checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    dataset_name = ""
    if args.coco_val.find("nyu") > 0:
        dataset_name = "nyu"
    elif args.coco_val.find("m3d") > 0:
        dataset_name = "m3d"
    elif args.coco_val.find("scannet") > 0:
        dataset_name = "scannet"
    tag = ""
    if args.mesh_depth:
        tag = "meshD_"
    else:
        tag = "holeD_"

    if args.refined_depth:
        tag += "refinedD"
    else:
        tag += "rawD"
    
    args.model_name = "bts_{}_{}".format(dataset_name, tag)


    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_folder = os.path.join(args.output_save_folder , "BTS_infer_{}".format(time_tag))
    os.makedirs(output_folder, exist_ok=True)
    
    print('Saving result pngs..')

    log_file_save_path = os.path.join(output_folder, "infer.log")
    logging.basicConfig(filename=log_file_save_path, filemode="a", level=logging.INFO, format="%(asctime)s %(name)s:%(levelname)s:%(message)s")
    logging.info("output folder {}".format(output_folder))
    logging.info("checkpoint {}".format(args.resume_checkpoint_path))

    mirror3d_eval = Mirror3d_eval(args.refined_depth,logger=logging, Input_tag="RGB", method_tag="BTS")
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader.data)):

            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())

            _, _, _, _, depth_est = model(image, focal)
            pred_depth = depth_est.cpu().numpy().squeeze()

            color_img_path = sample["image_path"][0]
        
            mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth,  args.depth_shift, color_img_path)
            mirror3d_eval.save_result(output_folder, pred_depth, args.depth_shift, color_img_path)

    mirror3d_eval.print_mirror3D_score()

    
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get Setting :D')

    parser.add_argument('--model_name',                type=str,   help='model name', default='bts')
    #TODO
    parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
    #TODO
    parser.add_argument('--mesh_depth',             type=bool,  help='using coco input format or not', default=True)
    # TODO
    parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/dt_only_15m3d_noraml/pos_test_normalFormat_15_normal.json')
    # TODO
    parser.add_argument('--coco_train',                type=str,   help='coco json path', default='/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/dt_only_15m3d_noraml/pos_train_normalFormat_15_normal.json')
    # TODO
    parser.add_argument('--coco_train_root',           type=str,   help='coco data root', 
        default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608")
    # TODO
    parser.add_argument('--coco_val_root',             type=str,   help='coco data root', 
        default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608")
    # TODO
    parser.add_argument('--coco_focal_len',            type=str,   help='focal length of input data; correspond to INPUT DEPTH!', default="519")
    # TODO 
    parser.add_argument('--depth_shift',               type=int,   help='depth shift to meter', default=4000) # 4000 for m3d
    # TODO if coda boom
    parser.add_argument('--input_height',              type=int,   help='input height', default=512) # 480
    # TODO  if coda boom
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640) # 640
    parser.add_argument('--coco_input',                            help='using coco input format or not',action='store_true')
    # TODO
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    # TODO
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    # TODO
    parser.add_argument('--resume_checkpoint_path',           type=str,   help='path to a checkpoint to load', default='/project/3dlg-hcvc/jiaqit/output/bts/bts_nyu_holeD_rawD_2020-12-27-15-43-03/checkpoint/model-14000')
    # TODO
    parser.add_argument('--output_save_folder',       type=str,   help='output_main_folder only use during inference', default='/project/3dlg-hcvc/jiaqit/exp_result')


    parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                        default='densenet161_bts')
    parser.add_argument('--data_path', type=str, help='path to the data', default="../dataset/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', default="train_test_inputs/nyudepthv2_test_files_with_gt.txt")
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
    parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
    args = parser.parse_args()

    args.coco_input = True # TODO 
    model_dir = os.path.dirname(args.resume_checkpoint_path)
    sys.path.append(model_dir)
    test(args)