import os
import torch
import numpy as np
import sys

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch

from detectron2.modeling import build_model

from mirror3d_lib.engine.defaults import Mirror3dTrainer
from mirror3d_lib.config.config import get_cfg
from mirror3d_lib.data.datasets.register_mirror3d_coco import register_mirror3d_coco_instances

import time
from contextlib import redirect_stdout
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):

    cfg = get_cfg() 
    cfg.merge_from_file(args.config)
    train_name = args.coco_train.split("/")[-1].split(".")[0]
    val_name = args.coco_val.split("/")[-1].split(".")[0]
    register_mirror3d_coco_instances(train_name, {}, args.coco_train, args.coco_train_root) 
    register_mirror3d_coco_instances(val_name, {}, args.coco_val, args.coco_val_root) 

    cfg.TRAIN_COCO_JSON = args.coco_train
    cfg.VAL_COCO_JSON = args.coco_val
    cfg.TRAIN_IMG_ROOT = args.coco_train_root
    cfg.VAL_IMG_ROOT = args.coco_val_root
    cfg.TRAIN_NAME = train_name
    cfg.VAL_NAME = val_name
    cfg.DATASETS.TRAIN = train_name
    cfg.DATASETS.TEST = val_name
    cfg.ANCHOR_NORMAL_CLASS_NUM = np.load(cfg.ANCHOR_NORMAL_NYP).shape[0]
    cfg.REFINED_DEPTH = args.refined_depth
    cfg.MESH_DEPTH = args.mesh_depth
    cfg.FOCAL_LENGTH = int(args.coco_focal_len)
    cfg.DEPTH_SHIFT = args.depth_shift
    cfg.EVAL_HEIGHT = args.input_height
    cfg.EVAL_WIDTH = args.input_width
    cfg.INPUT.MIN_SIZE_TRAIN = (args.input_height)
    cfg.INPUT.MIN_SIZE_TEST = args.input_height
    cfg.INPUT.MAX_SIZE_TRAIN = (args.input_width)
    cfg.INPUT.MAX_SIZE_TEST = args.input_width
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.num_epochs * 500
    cfg.SOLVER.STEPS = (int(args.num_epochs * 350), int(args.num_epochs * 400))
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_save_freq
    cfg.TEST.EVAL_PERIOD = args.checkpoint_save_freq
    cfg.MODEL.WEIGHTS = args.resume_checkpoint_path
    cfg.OUTPUT_DIR = args.log_directory
    
    if cfg.EVAL:
        eval_output_tag = ""
        if cfg.REF_DEPTH_TO_REFINE:
            eval_output_tag = cfg.REF_DEPTH_TO_REFINE.split("/")[-2]
        else:
            if not cfg.OBJECT_CLS:
                eval_output_tag = "ori_pr_preD_refD"
            elif cfg.DEPTH_EST and cfg.RGBD_INPUT:
                eval_output_tag = "mirror3d_preD_refD"
            else:
                eval_output_tag = "mirror3d_preD"
            eval_output_tag = eval_output_tag + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, eval_output_tag) + "_" + cfg.REF_DEPTH_TO_REFINE.split("/")[-1].split(".")[0] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = Mirror3dTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Mirror3dTrainer.test(cfg, model)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """

    if os.path.exists(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer = Mirror3dTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    yml_save_path = os.path.join(cfg.OUTPUT_DIR, "training_config.yml")
    with open(yml_save_path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    return trainer.train()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mirror3dnet/config/mirror3dnet_config.yml", type=str , help="path to config file")
    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # Input config (mirror3d)
    parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
    parser.add_argument('--mesh_depth',                action='store_true',  help='using coco input format or not')
    parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train',                type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train_root',           type=str,   help='coco data root', default="")
    parser.add_argument('--coco_val_root',             type=str,   help='coco data root', default="")

    # Data information config (mirror3d)
    parser.add_argument('--coco_focal_len',            type=str,   help='focal length of input data; correspond to INPUT DEPTH!', default="519") 
    parser.add_argument('--depth_shift',               type=int,   help='nyu : 1000, m3d : 4000', default=1000) 
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)   
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640) 

    # Network config (mirror3d)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=100)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--resume_checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

    # Log and save (mirror3d)
    parser.add_argument('--log_directory',             type=str,   help='training output folder', default='output')
    parser.add_argument('--checkpoint_save_freq',                 type=int,   help='Checkpoint saving frequency in global steps /iteration; nyu 5000; m3d 10000', default=500)

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    print("############## config file ###############", args.config)