from lib.core.config import cfg, merge_cfg_from_file, print_configs
from tools.parse_arg_train import TrainOptions
import sys
import datetime
import time
import os
from utils.Mirror3D_eval import Mirror3d_eval
import cv2
from tqdm import tqdm
import logging

train_opt = TrainOptions()
train_args = train_opt.parse()

time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

dataset_name = ""
if train_args.coco_train.find("nyu") > 0:
    dataset_name = "nyu"
elif train_args.coco_train.find("m3d") > 0:
    dataset_name = "m3d"
elif train_args.coco_train.find("scannet") > 0:
    dataset_name = "scannet"
tag = ""
if train_args.mesh_depth:
    tag = "meshD_"
else:
    tag = "holeD_"

if train_args.refined_depth:
    tag += "refinedD"
else:
    tag += "rawD"

train_opt.opt.model_name = "vnl_{}_{}".format(dataset_name, tag)
train_opt.opt.results_dir = os.path.join(train_opt.opt.results_dir, "vnl","{}_{}".format(train_opt.opt.model_name, time_tag))
os.makedirs(train_opt.opt.results_dir, exist_ok=True)
merge_cfg_from_file(train_args)
cfg.TRAIN.LOG_DIR = train_opt.opt.results_dir

from data.load_dataset import CustomerDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err
from lib.models.metric_depth_model import *

from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue
import math
import traceback

from tools.parse_arg_val import ValOptions
from lib.models.image_transfer import resize_image

logger = setup_logging(__name__)


def train(train_dataloader, model, epoch, loss_func,
          optimizer, scheduler, training_stats, val_dataloader=None, val_err=[], ignore_step=-1):
    """
    Train the model in steps
    """
    model.train()
    epoch_steps = math.ceil(len(train_dataloader) / cfg.TRAIN.batchsize)
    base_steps = epoch_steps * epoch + ignore_step if ignore_step != -1 else epoch_steps * epoch
    for i, data in enumerate(train_dataloader):
        if ignore_step != -1 and i > epoch_steps - ignore_step:
            return
        scheduler.step()  # decay lr every iteration
        training_stats.IterTic()
        try:
            out = model(data)
        except:
            print(data["A_paths"], data["B_paths"])
            continue
        losses = loss_func.criterion(out['b_fake_softmax'], out['b_fake_logit'], data, epoch)
        optimizer.optim(losses)
        step = base_steps + i + 1
        training_stats.UpdateIterStats(losses)
        training_stats.IterToc()
        training_stats.LogIterStats(step, epoch, optimizer.optimizer, val_err[0])

        # validate the model
        if step % cfg.TRAIN.VAL_STEP == 0 and step != 0 and val_dataloader is not None:
            model.eval()
            val_err[0] = val(val_dataloader, model, False)
            # training mode
            model.train()

        if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0:
            save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])
    


def val(val_dataloader, model, final_result):
    """
    Validate the model.
    """
    log_file_save_path = os.path.join(cfg.TRAIN.LOG_DIR, "exp_output.log")
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.basicConfig(filename=log_file_save_path, filemode="a", level=logging.INFO, format=FORMAT)
    logging.info("output folder {}".format(cfg.TRAIN.LOG_DIR))
    mirror3d_eval = Mirror3d_eval(train_args.refined_depth, logging, Input_tag="RGB", method_tag="VNL")

    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel}
    for i, data in enumerate(tqdm(val_dataloader)):
        # invalid_side = data['invalid_side'][0]
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake'])

        pred_depth = pred_depth * data['depth_shift'].cuda()
        pred_depth = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        smoothed_criteria = validate_err(pred_depth, data['B_raw'], smoothed_criteria, (45, 471, 41, 601))

        color_img_path = data["A_paths"][0]
        gt_depth_path = data["B_paths"][0]
        gt_depth = cv2.resize(cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
        pred_depth_scale = (pred_depth / pred_depth.max() *10000).astype(np.uint16)
        mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth_scale / data['depth_shift'], data['depth_shift'], color_img_path)
        if final_result:
            mirror3d_eval.save_result(cfg.TRAIN.LOG_DIR, pred_depth_scale / data['depth_shift'], data['depth_shift'], color_img_path)

        
    mirror3d_eval.print_mirror3D_score()
    print("update : {}".format(cfg.TRAIN.LOG_DIR))
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue()}


if __name__=='__main__':
    # Train args
    
    
    config_save_path = os.path.join(train_opt.opt.results_dir, "setting.txt")
    # train_opt.print_options(train_args)
    with open(config_save_path, "w") as file:
        file.write("####################### train args #######################")
        file.write("\n")
        for item in train_args.__dict__.items():
            file.write("--{} {}".format(item[0],item[1]))
            file.write("\n")
    print("output saved to : ", train_opt.opt.results_dir)
    print("config_save_path : ", config_save_path)


    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0
    # val_opt.print_options(val_args)

    with open(config_save_path, "w") as file:
        file.write("####################### val args #######################")
        file.write("\n")
        for item in val_args.__dict__.items():
            file.write("--{} {}".format(item[0],item[1]))
            file.write("\n")
    print("output saved to : ", val_args.results_dir)
    print("config_save_path : ", config_save_path)

    
    
    

    train_dataloader = CustomerDataLoader(train_args)
    train_datasize = len(train_dataloader)
    gpu_num = torch.cuda.device_count()
    

    val_dataloader = CustomerDataLoader(val_args)
    val_datasize = len(val_dataloader)

    # tensorboard logger
    os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)

    from tensorboardX import SummaryWriter
    tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # training status for logging
    training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,tblogger)

    # total iterations
    total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters
    cfg.TRAIN.GPU_NUM = gpu_num
    cfg.TRAIN.VAL_STEP = train_args.siter
    cfg.TRAIN.SNAPSHOT_ITERS = train_args.siter

    # load model
    model = MetricDepthModel()

    if gpu_num != -1:
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))
        model.cuda()

    optimizer = ModelOptimizer(model)
    loss_func = ModelLoss()

    val_err = [{'abs_rel': 0}]
    ignore_step = -1

    lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lr_optim_lambda)

    # load checkpoint
    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        ignore_step = train_args.start_step - train_args.start_epoch * math.ceil(train_datasize / train_args.batchsize)

    if gpu_num != -1:
        model = torch.nn.DataParallel(model)
    try:
        for epoch in range(train_args.start_epoch, train_args.epoch):
            # training
            train(train_dataloader, model, epoch, loss_func, optimizer, scheduler, training_stats,
                  val_dataloader, val_err, ignore_step)
            ignore_step = -1
        model.eval()    
        _ = val(val_dataloader, model, True)

    except (RuntimeError, KeyboardInterrupt):

        logger.info('Save ckpt on exception ...')
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard:
            tblogger.close()
