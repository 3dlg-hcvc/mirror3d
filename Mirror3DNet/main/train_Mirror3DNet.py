import os
import torch
import numpy as np
import sys
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from planrcnn_detectron2_lib.engine import DefaultTrainer, launch
from planrcnn_detectron2_lib.data.datasets import register_coco_instances
from planrcnn_detectron2_lib.config import get_cfg
from planrcnn_detectron2_lib.modeling import build_model
import time
from contextlib import redirect_stdout
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):

    cfg = get_cfg() # chris : get default configuration
    cfg.merge_from_file(args.config)
    tag = ""
    for train_idx in range(len(cfg.TRAIN_COCO_JSON)):
        register_coco_instances(cfg.TRAIN_NAME[train_idx], {}, cfg.TRAIN_COCO_JSON[train_idx], cfg.TRAIN_IMG_ROOT[train_idx]) 
        tag = tag + cfg.TRAIN_NAME[train_idx]
    for val_idx in range(len(cfg.VAL_COCO_JSON)):
        register_coco_instances(cfg.VAL_NAME[val_idx], {}, cfg.VAL_COCO_JSON[val_idx], cfg.VAL_IMG_ROOT[val_idx]) 

    cfg.DATASETS.TRAIN = cfg.TRAIN_NAME
    cfg.DATASETS.TEST = cfg.VAL_NAME
    cfg.ANCHOR_NORMAL_CLASS_NUM = np.load(cfg.ANCHOR_NORMAL_NYP).shape[0]


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
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DefaultTrainer.test(cfg, model)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """

    if os.path.exists(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    yml_save_path = os.path.join(cfg.OUTPUT_DIR, "training_config.yml")
    with open(yml_save_path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    return trainer.train()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nyu_mirror3d_config.yml", type=str , help="path to config file")
    parser.add_argument("--resume", default=True, type=bool)
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

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