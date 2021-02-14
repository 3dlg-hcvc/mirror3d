import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

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
from mirror3d_resnet import resnet50
from PIL import Image
import random
from tensorboardX import SummaryWriter
from classifier_Dataset import *
import logging
import datetime as d


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# TODO
parser.add_argument('--model_name', default="debug", type=str)
# TODO
parser.add_argument('--data_len', default=500, type=int) 
# TODO
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# TODO
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# TODO retrain
parser.add_argument('--resume', default='/project/3dlg-hcvc/jiaqit/output/classifier/checkpoint/epoch_20_checkpoint.pth.tar', type=str, metavar='PATH', # TODO checkpoint/checkpoint.pth.tar
                    help='path to latest checkpoint (default: none)')
# TODO 
parser.add_argument('-e', '--evaluate', action='store_true', type=bool) 
# TODO output checkpoint & log file save directory
parser.add_argument('--log_directory',    type=str,   help='training output folder', default='/project/3dlg-hcvc/jiaqit/output/classifier')
# TODO retrain
parser.add_argument('--checkpoint_save_freq',      type=int,   help="classifier's save frequncy (measure in EPOCH), usually set as 10", default=10)
# TODO if coda boom
parser.add_argument('--input_height',              type=int,   help='input height', default=512) # 480
# TODO  if coda boom
parser.add_argument('--input_width',               type=int,   help='input width',  default=640) # 640
# TODO 
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--train_pos_list',  # TODO retrain
        default="/local-scratch/share_data/scannet_extracted/info/classifier/positive/all.txt",
        type=str)
parser.add_argument('--train_neg_list', # TODO retrain
        default="/local-scratch/share_data/scannet_extracted/info/classifier/negative/training.txt",
        type=str)

parser.add_argument('--val_pos_list',  # TODO
        default="/local-scratch/share_data/scannet_extracted/info/classifier/positive/all.txt",
        type=str)
parser.add_argument('--val_neg_list',  # TODO
        default="/local-scratch/share_data/scannet_extracted/info/classifier/negative/training.txt",
        type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# TODO retrain
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
args = parser.parse_args()

best_acc1 = 0
run_start_time = d.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

checkpoint_save_folder = os.path.join(args.log_directory, "checkpoint")
os.makedirs(checkpoint_save_folder, exist_ok=True)

tb_writer_path = os.path.join(args.log_directory, "tb.log")
writer = SummaryWriter(tb_writer_path,filename_suffix="annotation_classifier")

log_file_save_path =  os.path.join(args.log_directory, "classifier.log")
logging.basicConfig(filename=log_file_save_path, filemode="a", level=logging.INFO, format="%(asctime)s %(name)s:%(levelname)s:%(message)s")
logging.info("output folder {}".format(args.log_directory))

def main():
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = resnet50()


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                transforms.Resize((args.input_height, args.input_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    # TODO change path
    train_dataset = Pos_Neg_eql_Dataset(args.train_pos_list, args.train_neg_list,train_transform) 
    
    val_transform = transforms.Compose([
                transforms.Resize((args.input_height, args.input_width)),
                transforms.ToTensor(),
                normalize,
            ])
    val_dataset = Pos_Neg_eql_Dataset(args.val_pos_list, args.val_neg_list,val_transform)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("data len : ",args.data_len , "train_loader len :", len(train_loader))
    if args.evaluate:
        val_acc, val_f_measure_0_3, val_f_measure_1, val_recall, val_precision = validate(val_loader, model, criterion, args) 
        print("validate : val_acc {} val_f_measure_0_3 {} val_f_measure_1 {} val_recall {} val_precision {}".format(val_acc, val_f_measure_0_3, val_f_measure_1, val_recall, val_precision))
        return

    print("tensorboard : ", tb_writer_path)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate per epoch
        val_acc, val_f_measure_0_3, val_f_measure_1, val_recall, val_precision = validate(val_loader, model, criterion, args) 
        print("validate : val_acc {} val_f_measure_0_3 {} val_f_measure_1 {} val_recall {} val_precision {}".format(val_acc, val_f_measure_0_3, val_f_measure_1, val_recall, val_precision))
        logging.info("validate : val_acc {} val_f_measure_0_3 {} val_f_measure_1 {} val_recall {} val_precision {}".format(val_acc, val_f_measure_0_3, val_f_measure_1, val_recall, val_precision))

        writer.add_scalars( "validation info",{
                    'val_acc': val_acc,
                    'val_f_measure_0_3': val_f_measure_0_3,
                    'val_f_measure_1' : val_f_measure_1,
                    'val_recall' : val_recall,
                    'val_precision' : val_precision
                }, epoch) 
        
        # save checkponit per checkpoint_save_freq
        if epoch > 0 and epoch % args.checkpoint_save_freq == 0:
            checkpoint_save_path = os.path.join(checkpoint_save_folder, "epoch_{}_checkpoint.pth.tar".format(epoch))
            torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint_save_path)
            print("tensorboard : ", tb_writer_path, "checkpoint saved path : ",checkpoint_save_path)

    # save final checkponit
    checkpoint_save_path = os.path.join(checkpoint_save_folder, "checkpoint_final.pth.tar")
    torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_save_path)
    print("FINAL tensorboard : ", tb_writer_path, "checkpoint saved path : ",checkpoint_save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):

    model.train()

    for i, (_, images, target) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        _, predicted = output.max(1)

        print( epoch*len(train_loader) + i, "batch mean loss : ", loss.item(), "correct num : ", (predicted == target).sum().item(),"pos_num : ", target.sum().item())

        acc1 = accuracy(output, target, topk=(1,))
        f_measure_0_3, f_measure_1, recall, precision = f_measure(predicted, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalars( "recall",{
                    'recall': recall.item()
                },  epoch*len(train_loader) + i) # i = iteration 
        
        writer.add_scalars( "precision",{
                    'precision': precision.item()
                },  epoch*len(train_loader) + i) # i = iteration 
        
        writer.add_scalars( "f_measure",{
                    'f_measure_0.3': f_measure_0_3.item(),
                    'f_measure_1': f_measure_1.item()
                },  epoch*len(train_loader) + i) # i = iteration 
        

        writer.add_scalars( "training loss",{
                    'training loss': loss.item()
                },  epoch*len(train_loader) + i) # i = iteration 

        writer.add_scalars( "training acc",{
                    'training acc1': acc1[0]
                },  epoch*len(train_loader) + i)



def validate(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()
    predicted_all = torch.tensor([]).cuda().long()
    target_all = torch.tensor([]).cuda().long()
    with torch.no_grad():
        for _, images, target in val_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            _, predicted = output.max(1)

            predicted_all = torch.cat((predicted_all, predicted),0)
            target_all = torch.cat((target_all, target),0)


    f_measure_0_3, f_measure_1, recall, precision = f_measure(predicted_all, target_all)
    acc = (predicted_all == target_all).sum().float() / float(len(target_all))
    return acc , f_measure_0_3, f_measure_1, recall, precision


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def f_measure(predicted_all, target_all):
    predicted_all = predicted_all.cpu().detach().numpy()
    target_all = target_all.cpu().detach().numpy()
    TP = np.logical_and((predicted_all==1),(target_all==1)).sum().astype(float)
    FP = np.logical_and((predicted_all==1),(target_all==0)).sum().astype(float)
    FN = np.logical_and((predicted_all==0),(target_all==1)).sum().astype(float)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    beta = 0.3
    f_measure_0_3 = ((1+beta**2)*precision*recall )/((beta**2) * precision + recall)

    beta = 1
    f_measure_1 = ((1+beta**2)*precision*recall )/((beta**2) * precision + recall)

    return f_measure_0_3, f_measure_1, recall, precision


if __name__ == '__main__':
    main()
