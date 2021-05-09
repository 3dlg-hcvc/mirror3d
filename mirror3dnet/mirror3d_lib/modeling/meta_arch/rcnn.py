# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeliFalgorng.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from ..depth_predict import Depth

__all__ = ["Mirror3d_GeneralizedRCNN"]

@META_ARCH_REGISTRY.register()
class Mirror3d_GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.depth_predictor = Depth()
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.anchor_normal_class_num = cfg.ANCHOR_NORMAL_CLASS_NUM
        self.depth_shift = cfg.DEPTH_SHIFT
        self.inverse_depth = cfg.INVERSE_DEPTH
        self.RGBD_INPUT = cfg.RGBD_INPUT
        self.DEPTH_EST = cfg.DEPTH_EST
        self.ANCHOR_CLS = cfg.ANCHOR_CLS
        self.ANCHOR_REG = cfg.ANCHOR_REG
        self.OBJECT_SEG = cfg.OBJECT_SEG
        self.OBJECT_CLS = cfg.OBJECT_CLS
        self.cfg =  cfg
        self.OBJECT_BBOX_REG = cfg.OBJECT_BBOX_REG

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs): # chris // run here
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`Mirror3d_DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # self.training = True #  debug
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if self.DEPTH_EST:
            gt_depths = [torch.clamp(x["depth_image"], min=1e-4).to(self.device) for x in batched_inputs] # !!! important check here normalize depth
        if self.inverse_depth:
            gt_depths = [1.0 / torch.clamp(gt_depth, min=1e-4) for gt_depth in gt_depths]
        
        if "instances" in batched_inputs[0]: #  input here 
            try:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            except:
                gt_instances = None
        else:
            gt_instances = None

        features = self.backbone(images.tensor)


        losses = {}

        if self.DEPTH_EST:
            depth_estimate_loss = self.depth_predictor(features, gt_depths, self.training) # changed 
            losses.update(depth_estimate_loss) 
    
        print("#########batched_inputs : ", batched_inputs[0]["img_path"],batched_inputs[0]["instances"]._fields["anchor_normal_residuals"], batched_inputs[0]["instances"]._fields["anchor_normal_classes"])
        print("#########batched_inputs : ", batched_inputs[1]["img_path"],batched_inputs[1]["instances"]._fields["anchor_normal_residuals"], batched_inputs[1]["instances"]._fields["anchor_normal_classes"])
        if self.proposal_generator: #! get 1000/ 2000 proposals + use 256 proposals among thousands of proposals for rpn_loss;
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances) 
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, self.anchor_normal_class_num)#  changed !!! proposals : 16*1000 , self.anchor_normal_class_num 
        if not self.ANCHOR_CLS:
            detector_losses.pop("anchor_cls")
        if not self.ANCHOR_REG:
            detector_losses.pop("anchor_residual_reg")
        if not self.OBJECT_CLS:
            detector_losses.pop("loss_cls")
        if not self.OBJECT_SEG:
            detector_losses.pop("loss_mask")
        if not self.OBJECT_BBOX_REG:
            detector_losses.pop("loss_box_reg")
        else:
            losses.update(proposal_losses)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)
        
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        img_name_list = []
        gt_bbox = []
        gt_depths = []

        if "file_name" in batched_inputs[0]:
            img_name_list = [x["file_name"] for x in batched_inputs]
        if "annotations" in batched_inputs[0]:
            gt_bbox = [torch.tensor(x["annotations"][0]["bbox"]).to(self.device) for x in batched_inputs] # !!! evalution final return
        if "depth_image" in batched_inputs[0]:
            gt_depths = [torch.clamp(x["depth_image"], min=1e-4).to(self.device) for x in batched_inputs]
            
        if self.DEPTH_EST:
            pred_depth_list = self.depth_predictor(features, gt_depths, self.training) # changed 
            if self.inverse_depth:
                pred_depth_list = [(1.0/pred_depth)*self.depth_shift for pred_depth in pred_depth_list]
            else:
                pred_depth_list = [pred_depth*self.depth_shift for pred_depth in pred_depth_list]
        else:
            pred_depth_list = []

        if not self.OBJECT_BBOX_REG :
            return None, pred_depth_list

        results = None
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None) # ! get 1000 / 2000 proposal
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results = self.roi_heads(images, features, proposals, None) #  get inferenced result # changed !! add gt here (1)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)


        if do_postprocess:
            return Mirror3d_GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes, self.cfg.MASK_ON_OTHER_SIZE), pred_depth_list 
        else:
            return results, pred_depth_list



    def preprocess_image(self, batched_inputs): 
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if self.RGBD_INPUT:
            noisy_depth_images = [x["noisy_depth_image"].to(self.device) for x in batched_inputs]
            rgbds = [torch.cat([noisy_depth_images[i].unsqueeze(0), images[i]], dim=0) for i in range(len(noisy_depth_images))]
            images = ImageList.from_tensors(rgbds, self.backbone.size_divisibility)
        else:
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mask_on_other_size):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            if mask_on_other_size:
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
            else:
                height = image_size[0]
                width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

