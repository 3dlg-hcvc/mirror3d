#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in planrcnn_detectron2_lib.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import numpy as np
import sys

import planrcnn_detectron2_lib.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from planrcnn_detectron2_lib.config import get_cfg
from planrcnn_detectron2_lib.data import MetadataCatalog
from planrcnn_detectron2_lib.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from planrcnn_detectron2_lib.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from planrcnn_detectron2_lib.modeling import GeneralizedRCNNWithTTA
from planrcnn_detectron2_lib.data.datasets import register_coco_instances
from planrcnn_detectron2_lib.data.datasets import register_coco_instances
from planrcnn_detectron2_lib.data import MetadataCatalog, DatasetCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2
from PIL import Image
from skimage import io,data
from detectron2.utils.visualizer import ColorMode
from planrcnn_detectron2_lib.config import get_cfg
import os
import torch
import shutil
from planrcnn_detectron2_lib.modeling import build_model
import time
from contextlib import redirect_stdout
import argparse
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=False,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("planrcnn_detectron2_lib.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res[0].items()})
        return res


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
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """

    if os.path.exists(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    yml_save_path = os.path.join(cfg.OUTPUT_DIR, "training_config.yml")
    with open(yml_save_path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    return trainer.train()





if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chris_config/m3d_mirror3d_config.yml", type=str , help="path to config file")
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