# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import logging
import torch
import itertools
import torch.utils.data
from collections import OrderedDict

from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
)
from detectron2.checkpoint import DetectionCheckpointer

from mirror3d_lib.data.dataset_mapper import Mirror3d_DatasetMapper
from detectron2.utils.registry import Registry
from detectron2.data import build_detection_train_loader, build_detection_test_loader


# TODO delete later

from detectron2.utils.logger import setup_logger



class Mirror3dTrainer(DefaultTrainer):



    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        train_mapper = Mirror3d_DatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper= train_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        test_mapper = Mirror3d_DatasetMapper(cfg, True)
        return build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)

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
        results = dict()
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

        cls.output_list = output_list
        return OrderedDict()

    @classmethod
    def get_output_list(cls):
        return cls.output_list


