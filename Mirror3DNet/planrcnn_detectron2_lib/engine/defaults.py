# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import planrcnn_detectron2_lib.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from planrcnn_detectron2_lib.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from planrcnn_detectron2_lib.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from planrcnn_detectron2_lib.modeling import build_model
from planrcnn_detectron2_lib.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from planrcnn_detectron2_lib.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from chris_eval_class import Chris_eval

from . import hooks
from .train_loop import SimpleTrainer

import time
import cv2
from planrcnn_detectron2_lib.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import numpy as np

__all__ = ["default_argument_parser", "default_setup", "DefaultPredictor", "DefaultTrainer"]


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
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
    return parser


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class DefaultTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it mades.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in planrcnn_detectron2_lib.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:

    .. code-block:: python

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it, load all checkpointables
        (eg. optimizer and scheduler) and update iteration counter.

        Otherwise, load the model specified by the config (skip all checkpointables) and start from
        the first iteration.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`planrcnn_detectron2_lib.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`planrcnn_detectron2_lib.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`planrcnn_detectron2_lib.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
If you want DefaultTrainer to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
"""
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        output_list = []
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            output_list= inference_on_dataset(model, data_loader, evaluator)
            chris_eval = Chris_eval(output_list, cfg)
            chris_eval.eval_main()


        return OrderedDict(), output_list # TODO_chris maybe need to change here

# def refine_depth(instance_mask, plane_parameter, np_depth, f):
#     # plane : ax + by + cd + d = 0
#     h, w = np_depth.shape
#     a, b, c = plane_parameter
#     offset = (np_depth * instance_mask).sum()/ instance_mask.sum()
#     py = np.where(instance_mask)[0].mean()
#     px = np.where(instance_mask)[1].mean()
#     x0 = (px - w/2) * (offset/ f)
#     y0 = (py- h/2) * (offset/ f)
#     d = -(a*x0 + b*y0 + c*offset)
#     for y in range(h):
#         for x in range(w):
#             if  instance_mask[y][x]:
#                 n = np.array([a, b, c])
#                 # plane function : ax + by + cz + d = 0 ---> x = 0 , y = 0 , c = -d/c
#                 V0 = np.array([0, 0, -d/c])
#                 P0 = np.array([0,0,0])
#                 P1 = np.array([(x - w/2), (y - h/2), f ])

#                 j = P0 - V0
#                 u = P1-P0
#                 N = -np.dot(n,j)
#                 D = np.dot(n,u)
#                 sI = N / D
#                 I = P0+ sI*u

#                 np_depth[y,x] = I[2]
#     return np_depth


# def debug_depth(gt, pred, img_path, instances):
    
#     img = cv2.imread(img_path) 
#     v = Visualizer(img[:, :, ::-1], # chris : init result visulizer
#         metadata=MetadataCatalog.get("s3d_mirror_val"), 
#         scale=0.5, 
#         instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#         )
#     v, colors = v.draw_instance_predictions(instances.to("cpu")) # chris : use result visualizer to show the result
#     output_img = v.get_image()[:, :, ::-1]
#     valid_gt = gt > 1e-4
#     pred[pred< 1e-4] = 1e-4
#     depth_diff = abs(1/gt - 1/pred) * valid_gt
#     depth_loss =depth_diff.sum()/valid_gt.sum()
#     from PIL import Image
#     output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
#     new_im = Image.new('RGB', (320*2, 240))
#     new_im.paste(Image.fromarray(output_img, 'RGB'), (0,0))
#     new_im.paste(Image.fromarray(depth_diff*255).resize((320,240)) , (320,0))

#     return new_im, "{:.2f}_{}_{}".format(depth_loss, img_path.split("/")[-3], img_path.split("/")[-1])


# def eval_plane_refine_depth(time_tag, output_list, cfg):


#     def compute_errors(gt, pred, mask): # gt and pred are in m

#         gt = gt/1000.0
#         pred = pred/1000.0

#         min_depth_eval = 1e-3
#         max_depth_eval = 10

#         pred[pred < min_depth_eval] = min_depth_eval
#         pred[pred > max_depth_eval] = max_depth_eval
#         pred[np.isinf(pred)] = max_depth_eval

#         gt[np.isinf(gt)] = 0
#         gt[np.isnan(gt)] = 0

        
#         valid_mask = np.logical_and(gt > min_depth_eval, gt < max_depth_eval)
#         scale = np.sum(pred[valid_mask]*gt[valid_mask])/np.sum(pred[valid_mask]**2)
#         valid_mask = np.logical_and(valid_mask, mask)

#         gt = gt[valid_mask]
#         pred = pred[valid_mask]
        
#         thresh = np.maximum((gt / pred), (pred / gt))
#         d1 = (thresh < 1.25).mean()
#         d2 = (thresh < 1.25 ** 2).mean()
#         d3 = (thresh < 1.25 ** 3).mean()

#         rmse = (gt - pred) ** 2
#         rmse = np.sqrt(rmse.mean())

#         rmse_log = (np.log(gt) - np.log(pred)) ** 2
#         rmse_log = np.sqrt(rmse_log.mean())

#         abs_rel = np.mean(np.abs(gt - pred) / gt)
#         sq_rel = np.mean(((gt - pred)**2) / gt)

#         err = np.log(pred) - np.log(gt)
#         silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

#         err = np.abs(np.log10(pred) - np.log10(gt))
#         log10 = np.mean(err)

#         scaled_rms = np.sqrt(((scale * pred-gt)**2).mean())

#         # RMSE = np.sqrt(np.sum((((pred_depth-gt)**2 )*valid_gt )/valid_gt.sum()))
#         return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, scaled_rms



#     def LSE_normalize_RMSE_mirror_area(gt, pred_depth, mask):
#         pred_depth=pred_depth.astype("float")
#         gt=gt.astype('float')
#         gt = gt
#         pred_depth = pred_depth
#         valid_gt = (gt > 1e-4)
#         valid_gt = valid_gt * mask
#         scale = np.sum(pred_depth*gt)/np.sum(pred_depth**2)
#         pred_depth = scale * pred_depth
#         RMSE = np.sqrt(np.sum((((pred_depth-gt)**2 )*valid_gt )/valid_gt.sum()))
#         return RMSE

#     if cfg.SAVE_DEPTH_IMG:
#         output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag)
#         os.makedirs(output_folder, exist_ok=True)
#         info_txt_save_path = os.path.join(output_folder, "mask_gtDepth_pDepth_npDepth.txt")
#         AR_correct_info_txt_save_path = os.path.join(output_folder, "AR_correct_mask_gtDepth_pDepth_npDepth.txt")
#         if os.path.exists(info_txt_save_path):
#             os.system('rm ' + info_txt_save_path)
#         if os.path.exists(AR_correct_info_txt_save_path):
#             os.system('rm ' + AR_correct_info_txt_save_path)
    
#     P_whole_RMSE_list = []
#     P_mirror_RMSE_list = []
#     P_none_mirror_RMSE_list = []
#     NP_whole_RMSE_list = []
#     NP_mirror_RMSE_list = []
#     NP_none_mirror_RMSE_list = []


#     Dt_correct_P_whole_RMSE_list = []
#     Dt_correct_P_mirror_RMSE_list = []
#     Dt_correct_P_none_mirror_RMSE_list = []
#     Dt_correct_NP_whole_RMSE_list = []
#     Dt_correct_NP_mirror_RMSE_list = []
#     Dt_correct_NP_none_mirror_RMSE_list = []
#     sample_num = len(output_list)


    

#     if cfg.DEPTH_EST:
#         num_test_samples = len(output_list)
#         silog = np.zeros(num_test_samples, np.float32)
#         log10 = np.zeros(num_test_samples, np.float32)
#         rms = np.zeros(num_test_samples, np.float32)
#         log_rms = np.zeros(num_test_samples, np.float32)
#         abs_rel = np.zeros(num_test_samples, np.float32)
#         sq_rel = np.zeros(num_test_samples, np.float32)
#         d1 = np.zeros(num_test_samples, np.float32)
#         d2 = np.zeros(num_test_samples, np.float32)
#         d3 = np.zeros(num_test_samples, np.float32)
#         scaled_rms = np.zeros(num_test_samples, np.float32)
#         for i, item in enumerate(output_list):
#             one_output, one_input = item
#             gt_depth = cv2.imread(one_input[0]["depth_path"], cv2.IMREAD_ANYDEPTH)
#             pred_depth = one_output[1][0].detach().cpu().numpy()
#             pred_depth[pred_depth<0] = 0
            
#             NP_whole_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, pred_depth/cfg.DEPTH_SHIFT, True))
#             silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i], scaled_rms[i] = compute_errors(gt_depth, pred_depth, True)
#             if cfg.SAVE_DEPTH_IMG:
#                 current_test_root = ""
#                 raw_input_img_path = one_input[0]["file_name"] 
#                 for one_test_img_root in cfg.TEST_IMG_ROOT:
#                     if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
#                         current_test_root = one_test_img_root
#                         break
#                 gt_depth_path = one_input[0]["depth_path"]
#                 raw_input_img_path = one_input[0]["file_name"]
#                 depth_np_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(raw_input_img_path, current_test_root)).replace("raw", "depth_np") )[0]
#                 os.makedirs(depth_np_save_folder, exist_ok=True)
#                 depth_np_save_path = os.path.join(depth_np_save_folder, raw_input_img_path.split("/")[-1])
#                 depth_np_save_path = depth_np_save_path.replace(".jpg",".png")
#                 cv2.imwrite(depth_np_save_path, pred_depth.astype(np.uint16))
#                 if os.path.exists(gt_depth_path):
#                     with open(info_txt_save_path, "a") as file:
#                         file.write("{} {} {} {}".format("None", gt_depth_path, "None", depth_np_save_path))
#                         file.write("\n")
#                     print("{} {} {} {}".format("None", gt_depth_path, "None", depth_np_save_path), " ----- info TXT : " ,info_txt_save_path)
#                 else:
#                     print("error some path not exist")
#                 sys.stdout.flush()

#         print("##################### Depth Estimation RMSE ####################")
#         print("cfg.OUTPUT_DIR : ",cfg.OUTPUT_DIR)
#         print("cfg.DEPTH_SHIFT : ", cfg.DEPTH_SHIFT)
#         print("--------------------{:20}--------------------".format("whole image"))
#         print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
#             'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10', "scaled_rms"))
#         print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
#             d1.mean(), d2.mean(), d3.mean(),
#         abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean(), scaled_rms.mean()))
#         # print("|{:40} | {:5f}|".format( "ori not refined whole img RMSE", np.mean(NP_whole_RMSE_list)))

#         if not cfg.EVAL:
#             storage = get_event_storage()
#             storage.put_scalar("DE whole_img scaled_rms",np.mean(scaled_rms.mean()))
#             storage.put_scalar("DE whole_img rms",np.mean(rms.mean()))
#             return

#     for one_output, one_input in output_list:
        
#         pred_depth = one_output[1][0].detach().cpu().numpy()
#         gt_depth = cv2.imread(one_input[0]["depth_path"], cv2.IMREAD_ANYDEPTH)
        


#         gt_mask = cv2.imread(one_input[0]["mask_path"],cv2.IMREAD_GRAYSCALE)
#         mirror_mask = gt_mask > 0
#         if not cfg.DEPTH_EST:
#             instances = one_output[0][0]["instances"]
#             _, pred_must_correct = draw_gt_bbox(one_input[0]["annotations"] ,gt_mask,instances.pred_anchor_classes)
#         np_pred_depth = pred_depth.copy()
        
#         # -------------- refine depth with predict anchor normal ------------
#         if cfg.ANCHOR_CLS:
#             anchor_normals = np.load(cfg.ANCHOR_NORMAL_NYP)
#             for instance_idx, pred_anchor_normal_class in enumerate(instances.pred_anchor_classes):
#                 instance_mask = instances.pred_masks[instance_idx].detach().cpu().numpy()
#                 z = (instance_mask*pred_depth).sum() / (instance_mask>0).sum()
#                 y = np.where(instance_mask>0)[0].mean() # h
#                 x = np.where(instance_mask>0)[1].mean() # w
#                 if pred_anchor_normal_class >= 3:
#                     continue
#                 else:
#                     if cfg.ANCHOR_REG:
#                         plane_normal = anchor_normals[pred_anchor_normal_class] + instances.pred_residuals[instance_idx].detach().cpu().numpy()
#                     else:
#                         plane_normal = anchor_normals[pred_anchor_normal_class]
#                 # plane_normal = one_input[0]["annotations"][0]["mirror_normal_camera"]
#                 # plane_normal = anchor_normals[one_input[0]["annotations"][0]["anchor_normal_class"]] + one_input[0]["annotations"][0]["anchor_normal_residual"]
#                 a, b, c = unit_vector(plane_normal)
                
#                 pred_depth = refine_depth(instance_mask, [a, b, c], pred_depth, cfg.FOCAL_LENGTH)        


        
#             current_test_root = ""
#             raw_input_img_path = one_input[0]["file_name"] 
#             for one_test_img_root in cfg.TEST_IMG_ROOT:
#                 if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
#                     current_test_root = one_test_img_root
#                     break
            
#         np_pred_depth = np_pred_depth.astype(np.uint16)
#         depth_p = pred_depth.astype(np.uint16)

#         # P_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, mirror_mask))
#         # P_none_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, mirror_mask == False))
#         # P_whole_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, True))
#         # NP_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, mirror_mask))
#         # NP_none_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, mirror_mask == False))
#         # NP_whole_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, True))

#         # if pred_must_correct:
#         #     Dt_correct_P_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, mirror_mask))
#         #     Dt_correct_P_none_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, mirror_mask == False))
#         #     Dt_correct_P_whole_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, depth_p/cfg.DEPTH_SHIFT, True))
#         #     Dt_correct_NP_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, mirror_mask))
#         #     Dt_correct_NP_none_mirror_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, mirror_mask == False))
#         #     Dt_correct_NP_whole_RMSE_list.append(LSE_normalize_RMSE_mirror_area(gt_depth/cfg.DEPTH_SHIFT, np_pred_depth/cfg.DEPTH_SHIFT, True))

#         if cfg.SAVE_DEPTH_IMG:
#             gt_mask_path = one_input[0]["mask_path"]
#             gt_depth_path = one_input[0]["depth_path"]
#             depth_np_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(raw_input_img_path, current_test_root)).replace("raw", "depth_np") )[0]
#             depth_np_save_path = os.path.join(depth_np_save_folder, raw_input_img_path.split("/")[-1])
#             depth_p_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(raw_input_img_path, current_test_root)).replace("raw", "depth_p") )[0]
#             depth_p_save_path = os.path.join(depth_p_save_folder, raw_input_img_path.split("/")[-1])
#             os.makedirs(depth_np_save_folder, exist_ok=True)
#             os.makedirs(depth_p_save_folder, exist_ok=True)
#             cv2.imwrite(depth_np_save_path, np_pred_depth)
#             cv2.imwrite(depth_p_save_path, depth_p)
#             if os.path.exists(gt_mask_path) and os.path.exists(gt_depth_path) and os.path.exists(depth_p_save_path) and os.path.exists(depth_np_save_path):
#                 with open(info_txt_save_path, "a") as file:
#                     file.write("{} {} {} {}".format(gt_mask_path, gt_depth_path, depth_p_save_path, depth_np_save_path))
#                     file.write("\n")
#                 print("{} {} {} {}".format(gt_mask_path, gt_depth_path, depth_p_save_path, depth_np_save_path), " ----- info TXT : " ,info_txt_save_path)
#                 # if pred_must_correct:
#                 #     with open(AR_correct_info_txt_save_path, "a") as file:
#                 #         file.write("{} {} {} {}".format(gt_mask_path, gt_depth_path, depth_p_save_path, depth_np_save_path))
#                 #         file.write("\n")
#                 #     print("{} {} {} {}".format(gt_mask_path, gt_depth_path, depth_p_save_path, depth_np_save_path), " ----- info TXT : " ,AR_correct_info_txt_save_path)
#             else:
#                 print("error some path not exist")
#             sys.stdout.flush()
    
    

#     # print("##################### Depth Estimation RMSE ####################")
#     # print("|{:40} | {:5f}|".format( "plane refined whole img RMSE", np.mean(P_whole_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "plane refined mirror area RMSE", np.mean(P_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "plane refined none-mirror area RMSE", np.mean(P_none_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined whole img RMSE", np.mean(NP_whole_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined mirror area RMSE", np.mean(NP_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined none-mirror area RMSE", np.mean(NP_none_mirror_RMSE_list)))

#     # if not cfg.EVAL:
#     #     storage = get_event_storage()
#     #     storage.put_scalar("DE whole_img RMSE",np.mean(NP_whole_RMSE_list))

#     # print("##################### Depth Estimation RMSE one instance AC correct ####################")
#     # print("|{:40} | {:5f}|".format( "plane refined whole img RMSE", np.mean(Dt_correct_P_whole_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "plane refined mirror area RMSE", np.mean(Dt_correct_P_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "plane refined none-mirror area RMSE", np.mean(Dt_correct_P_none_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined whole img RMSE", np.mean(Dt_correct_NP_whole_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined mirror area RMSE", np.mean(Dt_correct_NP_mirror_RMSE_list)))
#     # print("|{:40} | {:5f}|".format( "ori not refined none-mirror area RMSE", np.mean(Dt_correct_NP_none_mirror_RMSE_list)))


# def get_dt_score(time_tag, output_list, cfg):
#     output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag)
#     os.makedirs(output_folder, exist_ok=True)
#     print(len(output_list))
#     estimate_fail = 0
#     for one_output, one_input in output_list:
#         instances = one_output[0][0]["instances"]
#         img_path = one_input[0]["file_name"]
#         if instances.pred_boxes.tensor.shape[0] <= 0:
#             print("######## no detection :", img_path)
#             continue

#         img = cv2.imread(img_path) 
#         v = Visualizer(img[:, :, ::-1], # chris : init result visulizer
#             metadata=MetadataCatalog.get(""), # TODO 
#             scale=0.5, 
#             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#             )
#         v, colors = v.draw_instance_predictions(instances.to("cpu")) # chris : use result visualizer to show the result
        
#         output_img = v.get_image()[:, :, ::-1]
#         output_img, predict_correct = draw_gt_bbox(one_input[0]["annotations"] ,output_img,instances.pred_anchor_classes)


#     print("sample may fail : " ,estimate_fail , "sample must corret : ", len(output_list) - estimate_fail)
#     print("##################### main output folder #################### {}".format(output_folder))

   
# def visualize_instance_chris(time_tag, output_list, cfg):
#     output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag)
#     os.makedirs(output_folder, exist_ok=True)
#     output_json_save_path = os.path.join(output_folder, "output_info.json")
#     output_info = dict()
#     print(len(output_list))
#     estimate_fail = 0
#     for one_output, one_input in output_list:
        
#         instances = one_output[0][0]["instances"]
#         img_path = one_input[0]["file_name"]
#         if instances.pred_boxes.tensor.shape[0] <= 0:
#             print("######## no detection :", img_path)
#             continue

#         img = cv2.imread(img_path) 
#         v = Visualizer(img[:, :, ::-1], # chris : init result visulizer
#             metadata=MetadataCatalog.get("s3d_mirror_val"), 
#             scale=0.5, 
#             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#             )
#         v, colors = v.draw_instance_predictions(instances.to("cpu")) # chris : use result visualizer to show the result
        
#         output_img = v.get_image()[:, :, ::-1]
#         output_img, predict_correct = draw_gt_bbox(one_input[0]["annotations"] ,output_img,instances.pred_anchor_classes)

#         raw_input_img_path = one_input[0]["file_name"] 
#         for one_test_img_root in cfg.TEST_IMG_ROOT:
#                 if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
#                     current_test_root = one_test_img_root
#                     break
        

#         if predict_correct:
#             correct_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "success_masked_img") )[0]
#             os.makedirs(correct_save_folder, exist_ok=True)
#             img_save_path = os.path.join(correct_save_folder, img_path.split("/")[-1])
#             cv2.imwrite(img_save_path, output_img)
#             print(  "mask debug image saved to :", img_save_path )
#             if cfg.ANCHOR_CLS and cfg.ANCHOR_REG:
#                 noraml_vis_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "success_noraml_vis") )[0]
#                 os.makedirs(noraml_vis_save_folder, exist_ok=True)
#                 noraml_vis_save_path = os.path.join(noraml_vis_save_folder, img_path.split("/")[-1])
#                 normal_vis_image = get_normal_vis(cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)
#         else:
#             false_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "fail_masked_img") )[0]

#             os.makedirs(false_save_folder, exist_ok=True)
#             img_save_path = os.path.join(false_save_folder, img_path.split("/")[-1])
#             cv2.imwrite(img_save_path, output_img)
#             print(  "mask debug image saved to :", img_save_path )
#             estimate_fail += 1
#             if cfg.ANCHOR_CLS and cfg.ANCHOR_REG:
#                 noraml_vis_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "fail_noraml_vis") )[0]
#                 os.makedirs(noraml_vis_save_folder, exist_ok=True)
#                 noraml_vis_save_path = os.path.join(noraml_vis_save_folder, img_path.split("/")[-1])
#                 normal_vis_image = get_normal_vis(cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)



#         # one_info = dict()
#         # one_info["gt_depth_path"] = one_input[0]["depth_path"]
#         # one_info["gt_mask_path"] = one_input[0]["mask_path"]
#         # one_info["gt_id_normal"] = dict()
#         # for item in one_input[0]:

        

#         # output_info[one_input[0]["depth_path"]] 


#     print("sample may fail : " ,estimate_fail , "sample must corret : ", len(output_list) - estimate_fail)
#     print("##################### main output folder #################### {}".format(output_folder))


# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)

# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::

#             >>> angle_between((1, 0, 0), (0, 1, 0))
#             1.5707963267948966
#             >>> angle_between((1, 0, 0), (1, 0, 0))
#             0.0
#             >>> angle_between((1, 0, 0), (-1, 0, 0))
#             3.141592653589793
#     """
#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
