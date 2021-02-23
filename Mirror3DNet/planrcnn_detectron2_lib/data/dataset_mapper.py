# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from . import detection_utils as utils
from . import transforms as T
import cv2

from planrcnn_detectron2_lib.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.anchor_normals = np.load(cfg.ANCHOR_NORMAL_NYP)
        self.REFINED_DEPTH = cfg.REFINED_DEPTH
        self.mesh_depth = cfg.MESH_DEPTH
        self.depth_shift = cfg.DEPTH_SHIFT
        self.RGBD_INPUT = cfg.RGBD_INPUT

        if cfg.UNIT_ANCHOR_NORMAL:
            for i in range(len(self.anchor_normals)):
                self.anchor_normals[i] = self.anchor_normals[i]/ (np.sqrt(self.anchor_normals[i][0]**2 + self.anchor_normals[i][1]**2 + self.anchor_normals[i][2]**2))
            

        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["img_path"], format=self.img_format) # changed : read depth image here

        if self.mesh_depth:
            if "mesh_refined_path" in dataset_dict and self.REFINED_DEPTH:
                depth_image = cv2.imread(dataset_dict["mesh_refined_path"], cv2.IMREAD_ANYDEPTH) / self.depth_shift
                depth_image, _ = T.apply_transform_gens(self.tfm_gens, depth_image.astype(np.float32))
            elif "mesh_raw_path" in dataset_dict and not self.REFINED_DEPTH:
                depth_image = cv2.imread(dataset_dict["mesh_raw_path"], cv2.IMREAD_ANYDEPTH) / self.depth_shift
                depth_image, _ = T.apply_transform_gens(self.tfm_gens, depth_image.astype(np.float32))
        else:
            if "hole_refined_path" in dataset_dict and self.REFINED_DEPTH:
                depth_image = cv2.imread(dataset_dict["hole_refined_path"], cv2.IMREAD_ANYDEPTH) / self.depth_shift
                depth_image, _ = T.apply_transform_gens(self.tfm_gens, depth_image.astype(np.float32))
            elif "hole_raw_path" in dataset_dict and not self.REFINED_DEPTH:
                depth_image = cv2.imread(dataset_dict["hole_raw_path"], cv2.IMREAD_ANYDEPTH) / self.depth_shift 
                depth_image, _ = T.apply_transform_gens(self.tfm_gens, depth_image.astype(np.float32))

        noisy_depth_image = []
        if "hole_raw_path" in dataset_dict and self.RGBD_INPUT:
            noisy_depth_image = cv2.imread(dataset_dict["hole_raw_path"], cv2.IMREAD_ANYDEPTH) / self.depth_shift
            noisy_depth_image, _ = T.apply_transform_gens(self.tfm_gens, noisy_depth_image.astype(np.float32))

        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            
 
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["depth_image"] = torch.as_tensor(depth_image.astype(np.float32))
        if noisy_depth_image and self.RGBD_INPUT:
            noisy_depth_image[noisy_depth_image<0] = 0
            dataset_dict["noisy_depth_image"] = torch.as_tensor(noisy_depth_image.astype(np.float32)) # changed !!!
        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals: # chris ： self.load_proposals = False
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None) # chris changed 
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices, anchor_normals=self.anchor_normals
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances( 
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)  
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict
