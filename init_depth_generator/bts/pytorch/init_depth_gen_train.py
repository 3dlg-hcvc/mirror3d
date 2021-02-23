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

import time
import argparse
import datetime
import sys
import os
from bts import *
import logging
import cv2

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from bts import BtsModel
from bts_dataloader import *
from utils.Mirror3D_eval import Mirror3d_eval

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='bts')
#TODO
parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
#TODO
parser.add_argument('--mesh_depth',                action='store_true',  help='using coco input format or not')
# TODO
parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json')
# TODO
parser.add_argument('--coco_train',                type=str,   help='coco json path', default='/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json')
# TODO
parser.add_argument('--coco_train_root',           type=str,   help='coco data root', 
    default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise")
# TODO
parser.add_argument('--coco_val_root',             type=str,   help='coco data root', 
    default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise")
# TODO
parser.add_argument('--coco_focal_len',            type=str,   help='focal length of input data; correspond to INPUT DEPTH!', default="310")
# TODO 
parser.add_argument('--depth_shift',               type=int,   help='nyu : 1000, m3d : 4000', default=1000) # 4000 for m3d
# TODO if coda boom
parser.add_argument('--input_height',              type=int,   help='input height', default=256) # 480
# TODO  if coda boom
parser.add_argument('--input_width',               type=int,   help='input width',  default=320) # 640
# TODO
parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
# TODO
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
# TODO
parser.add_argument('--resume_checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
# TODO
parser.add_argument('--checkpoint_save_freq',                 type=int,   help='Checkpoint saving frequency in global steps /iteration; nyu 5000; m3d 10000', default=500)
# TODO
# Log and save
parser.add_argument('--log_directory',             type=str,   help='training output folder', default='/project/3dlg-hcvc/jiaqit/output')
# TODO
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=1)


parser.add_argument('--coco_input',                type=bool,  help='using coco input format or not', default=True)
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                            default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default="../../dataset/nyu_depth_v2/sync/")
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', default="../../dataset/nyu_depth_v2/sync/")

parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default="../train_test_inputs/nyudepthv2_train_files_with_gt.txt")
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--tensorboard_log_freq',                  type=int,   help='Logging frequency in global steps', default=100)


# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with resume_checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation', default=False)
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',            type=bool,  help='if set, perform online eval in every checkpoint_save_freq steps', default=True)
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', default="../../dataset/nyu_depth_v2/official_splits/test/")
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', default="../../dataset/nyu_depth_v2/official_splits/test/")
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', default="../train_test_inputs/nyudepthv2_test_files_with_gt.txt")
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--eigen_crop',                type=bool,  help='if set, crops according to Eigen NIPS14', default=True)
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')


args = parser.parse_args()



inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd125', 'd125_2', 'd125_3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d125 = (thresh < 1.25).mean()
    d125_2 = (thresh < 1.25 ** 2).mean()
    d125_3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d125, d125_2, d125_3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False

def online_eval(model, logger, dataloader_eval, gpu, ngpus, args, final_result):

    mirror3d_eval = Mirror3d_eval(args.refined_depth, logger=logger,Input_tag="RGB", method_tag="BTS")
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            _, _, _, _, pred_depth = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth, args.depth_shift, eval_sample_batched["image_path"][0])

        if final_result:
            mirror3d_eval.save_result(args.log_directory, pred_depth, args.depth_shift, eval_sample_batched["image_path"][0])


    if not args.multiprocessing_distributed or gpu == 0:
        mirror3d_eval.print_mirror3D_score()
    return


def main_worker(gpu, ngpus_per_node, args):
    log_file_save_path = os.path.join(args.log_directory, "exp_output.log")
    
    print("log_file_save_path : {}".format(log_file_save_path))
    logging.basicConfig(filename=log_file_save_path, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)
    logging.info("log_file_save_path : {}".format(log_file_save_path))
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = BtsModel(args)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=[args.gpu])
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    temp_lower = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    temp_higher = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.encoder.parameters(), 'weight_decay': args.weight_decay},
                                   {'params': model.module.decoder.parameters(), 'weight_decay': 0}],
                                  lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.resume_checkpoint_path != '':
        if os.path.isfile(args.resume_checkpoint_path):
            print("Loading checkpoint '{}'".format(args.resume_checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.resume_checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume_checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.resume_checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.resume_checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval = BtsDataLoader(args, 'online_eval')

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        train_writer = SummaryWriter(args.log_directory , flush_secs=30)
        if args.do_online_eval:
            eval_summary_path = args.log_directory
            train_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                logging.info('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            # write to tensorbaord
            if global_step and global_step % args.tensorboard_log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.tensorboard_log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))
                logging.info(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    train_writer.add_scalar('silog_loss', loss, global_step)
                    train_writer.add_scalar('learning_rate', current_lr, global_step)
                    train_writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)

            checkpoint_save_folder = os.path.join(args.log_directory, "checkpoint")
            os.makedirs(checkpoint_save_folder,exist_ok=True)

            if  args.do_online_eval and global_step and global_step % args.checkpoint_save_freq == 0 and not model_just_loaded:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    model_save_name = 'model-{}'.format(global_step)
                    torch.save(checkpoint, os.path.join(checkpoint_save_folder , model_save_name))
                    logging.info("checkpoint saved to : {}".format(os.path.join(checkpoint_save_folder , model_save_name)))
                time.sleep(0.1)
                model.eval()
                online_eval(model, logging, dataloader_eval, gpu, ngpus_per_node, args, False)
                model.train()
                block_print()
                set_misc(model)
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    ################ saved all result after training ################
    online_eval(model, logging, dataloader_eval, gpu, ngpus_per_node, args, True)


def main():
    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    dataset_name = ""
    if args.coco_train.find("nyu") > 0:
        dataset_name = "nyu"
    elif args.coco_train.find("m3d") > 0:
        dataset_name = "m3d"
    elif args.coco_train.find("scannet") > 0:
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

    args.log_directory = os.path.join(args.log_directory, "bts","{}_{}".format(args.model_name, time_tag))
    os.makedirs(args.log_directory, exist_ok=True)
    config_save_path = os.path.join(args.log_directory, "setting.txt")

    with open(config_save_path, "w") as file:
        for item in args.__dict__.items():
            file.write("--{} {}".format(item[0],item[1]))
            file.write("\n")
    print("args saved to :", config_save_path)
    
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1
    

    if args.resume_checkpoint_path == '':
        pass
    else:
        loaded_model_dir = os.path.dirname(args.resume_checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'



    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every checkpoint_save_freq {} steps and save best models for individual eval metrics."
              .format(args.checkpoint_save_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':

    main()
