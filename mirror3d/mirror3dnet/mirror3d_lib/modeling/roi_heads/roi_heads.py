# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads

from mirror3d_lib.modeling.roi_heads.fast_rcnn import Mirror3d_FastRCNNOutputLayers

logger = logging.getLogger(__name__)





@ROI_HEADS_REGISTRY.register()
class Mirror3d_StandardROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        anchor_normal_class_num:int=None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        self.anchor_normal_class_num = anchor_normal_class_num #   changed
        if self.training:
            assert targets # add gt_anchor_normal_class gt_anchor_normal_residual to Instance in Proposal
            proposals = self.label_and_sample_proposals(proposals, targets) # proposals[0]._fields["gt_classes"] : [0, num_classes) or the background (num_classes).
        del targets 

        if self.training:
            losses = self._forward_box(features, proposals) # changed !!! add gt_acnhor *** here (1)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else: 
            pred_instances = self._forward_box(features, proposals) # changed !!! add gt_acnhor *** here (1.2)
            
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.

            pred_instances = self.forward_with_given_boxes(features, pred_instances) 
            return pred_instances

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = Mirror3d_FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances],
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]: #changed !!! add gt_anchor & anchor_prodiction here (2.1)
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features) 
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals) 
            return losses
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals) 
            return pred_instances


    def _sample_proposals_chris(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, anchor_normal_classes : torch.Tensor, anchor_normal_residuals : torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Based on the matching between N proposals and M groundtruth,
            sample the proposals and set their classification labels.

            Args:
                matched_idxs (Tensor): a vector of length N, each is the best-matched
                    gt index in [0, M) for each proposal.
                matched_labels (Tensor): a vector of length N, the matcher's label
                    (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
                gt_classes (Tensor): a vector of length M.

            Returns:
                Tensor: a vector of indices of sampled proposals. Each is in [0, N).
                Tensor: a vector of the same length, the classification label for
                    each sampled proposal. Each sample is labeled as either a category in
                    [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0 #  gt_classes.shape torch.Size([<number_of_instance_in_the_image>])
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs] # gt_classes.shape = torch.Size([len(matched_idxs)]); get the gt_class of intance that match the corresponding proposal
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes); previous the id_map set the gt_classes start from 0
            gt_classes[matched_labels == 0] = self.num_classes # matched_labels == 0 means the IOU is between the [lower_bound, upper_bound]
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            #  get gt_anchor_normal_classes
            anchor_normal_classes = anchor_normal_classes[matched_idxs]
            anchor_normal_classes[matched_labels == 0] = self.anchor_normal_class_num
            anchor_normal_classes[matched_labels == -1] = -1
            anchor_normal_residuals = anchor_normal_residuals[matched_idxs]
            anchor_normal_residuals[matched_labels == 0] = 0 # background
            anchor_normal_residuals[matched_labels == -1] = -1 # negative
      
        else: # if don't have gt instance in this image, all proposal should be background
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            anchor_normal_classes = torch.zeros_like(matched_idxs) + self.anchor_normal_class_num
            anchor_normal_residuals = torch.zeros([matched_idxs.shape[0],0]).shape
        # chris: now gt_classes = self.num_classes -> bg ; self.num_classes = -1 ignored; gt_classes = [0, self.num_classes) --> matched positive sample
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )   
            


        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs], anchor_normal_classes[sampled_idxs], anchor_normal_residuals[sampled_idxs] # gt_classes : [0, num_classes) or the background (num_classes).


    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0 #  gt_classes.shape torch.Size([<number_of_instance_in_the_image>])
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs] # gt_classes.shape = torch.Size([len(matched_idxs)]); get the gt_class of intance that match the corresponding proposal
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes); previous the id_map set the gt_classes start from 0
            gt_classes[matched_labels == 0] = self.num_classes # matched_labels == 0 means the IOU is between the [lower_bound, upper_bound]
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else: # if don't have gt instance in this image, all proposal should be background
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        # chris: now gt_classes = self.num_classes -> bg ; self.num_classes = -1 ignored; gt_classes = [0, self.num_classes) --> matched positive sample
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs] # gt_classes : [0, num_classes) or the background (num_classes).

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets): # chris // get gt_box 
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) 
            # sampled_idxs, gt_classes = self._sample_proposals( # gt_classes : [0, num_classes) or the background (num_classes).
            #     matched_idxs, matched_labels, targets_per_image.gt_classes #  get 128 outof ~1000 sample
            # )
            if "anchor_normal_classes" in targets_per_image._fields and "anchor_normal_residuals" in targets_per_image._fields:
                sampled_idxs, gt_classes, gt_anchor_normal_classes, gt_anchor_normal_residuals = self._sample_proposals_chris( # gt_classes : [0, num_classes) or the background (num_classes).
                    matched_idxs, matched_labels, targets_per_image.gt_classes, targets_per_image.anchor_normal_classes, targets_per_image.anchor_normal_residuals #  get 128 outof ~1000 sample
                )
            else:
                sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)
                gt_anchor_normal_classes = 0 
            
            
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes #  gt_classes : [0, num_classes) or the background (num_classes).
            
            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            if "anchor_normal_classes" in targets_per_image._fields and "anchor_normal_residuals" in targets_per_image._fields:
                try:
                    proposals_per_image._fields["gt_anchor_normal_classes"] = gt_anchor_normal_classes
                    proposals_per_image._fields["gt_anchor_normal_residuals"] = gt_anchor_normal_residuals
                except:
                    print("no gt_anchor_normal_classes -------------")
                    pass

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            

            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        return proposals_with_gt # proposals_with_gt[0]._fields["gt_classes"] : [0, num_classes) or the background (num_classes).

