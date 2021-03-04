# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import sys

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import numpy as np

__all__ = ["mirror3d_fast_rcnn_inference", "Mirror3d_FastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def mirror3d_fast_rcnn_inference(boxes, scores, anchor_scores, residuals,image_shapes, score_thresh, nms_thresh, topk_per_image,OBJECT_CLS):
    """
    Call `mirror3d_fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`Mirror3d_FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`Mirror3d_FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        mirror3d_fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image,anchor_score_per_image, residual_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, OBJECT_CLS # changed !! add anchor info here (3)
        )
        for scores_per_image, boxes_per_image, anchor_score_per_image, residual_per_image,image_shape in zip(scores, boxes, anchor_scores, residuals,image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image] # 2 , 3 anchor_cls_ce_loss ,ANCHOR_REGidual_mse_loss



def mirror3d_fast_rcnn_inference_single_image(
    boxes, scores, anchor_scores, pred_residual,image_shape, score_thresh, nms_thresh, topk_per_image, OBJECT_CLS
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `mirror3d_fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `mirror3d_fast_rcnn_inference`, but for only one image.
    """
    anchor_idx = anchor_scores.argmax(dim=1)
    pred_residual_3 = torch.stack([pred_residual[i][idx*3:(idx+1)*3] for i, idx in enumerate(anchor_idx)],0)

    if OBJECT_CLS:
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    else:
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(anchor_scores).all(dim=1)
    
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        anchor_scores = [valid_mask]
    
    if OBJECT_CLS:
        scores = scores[:, :-1]
    else:
        max_anchor_score_id = anchor_scores[:,:-1].argmax(dim=1)
        scores = torch.tensor([anchor_scores[id][a]  for id, a in enumerate(max_anchor_score_id)]).unsqueeze(dim=1).cuda() 

    
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    anchor_scores_filter_mask = anchor_scores[:, :-1] > score_thresh
    
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    anchor_scores = torch.tensor([anchor_scores[id][a]  for id, a in enumerate(anchor_idx)]).unsqueeze(dim=1).cuda()
    if filter_mask.shape[1] == 1:
        anchor_scores = anchor_scores[filter_mask]
        anchor_idx = anchor_idx.unsqueeze(dim=1)[filter_mask]
        pred_residual_3 = pred_residual_3[(filter_mask).nonzero()[:,0]]
    
    
    boxes, scores, filter_inds , anchor_idx, pred_residual_3, anchor_scores = boxes[keep], scores[keep], filter_inds[keep],anchor_idx[keep], pred_residual_3[keep], anchor_scores[keep]

    sys.stdout.flush()
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.anchor_scores = anchor_scores
    result.pred_anchor_classes = anchor_idx
    result.pred_residuals = pred_residual_3
    result.pred_classes = filter_inds[:, 1]
    
    return result, filter_inds[:, 0]


def cross_entropy(Y, P):
    import numpy as np
    Y = np.float_(Y)
    P = np.float_(P)

    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

class Mirror3d_FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """
    # changed 
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0,
        anchor_normal_score=None, 
        anchor_residual_pred=None,
        anchor_normals=None,
        ANCHOR_REG_method=None,
        anchor_cls=False,
        ANCHOR_REG=False,
        is_training=True,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.anchor_normals = anchor_normals
        self.ANCHOR_REG_method = ANCHOR_REG_method
        
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.image_shapes = [x.image_size for x in proposals]

        # changed !!!
        self.anchor_normal_score = anchor_normal_score
        self.anchor_residual_pred = anchor_residual_pred
        self.is_training = is_training
        self.anchor_cls = anchor_cls
        self.ANCHOR_REG = ANCHOR_REG

        # changed !!! : add gt *** to self. here (4)
        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals]) # torch.Size([128, 4])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0) # torch.Size([128])
            if proposals[0].has("gt_anchor_normal_classes"):
                self.gt_anchor_normal_classes = cat([p.gt_anchor_normal_classes for p in proposals], dim=0) # torch.Size([128])
            if proposals[0].has("gt_anchor_normal_residuals"):
                self.gt_anchor_normal_residuals = cat([p.gt_anchor_normal_residuals for p in proposals], dim=0) # torch.Size([128])
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        


        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        fg_score_test = self.pred_class_logits[:, :-1]
        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean") # self.pred_class_logits.shape torch.Size([128, 2]) self.gt_classes  torch.Size([128])
    
    def softmax_cross_entropy_loss_anchor_normal(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """

        if self._no_instances or not self.anchor_cls:
            return 0.0 * self.anchor_normal_score.sum()
        else:
            if self.is_training:
                self._log_accuracy() #  torch.Size([128, 8]) & torch.Size([128])
            return F.cross_entropy(self.anchor_normal_score, self.gt_anchor_normal_classes, reduction="mean") # self.pred_class_logits.shape torch.Size([128, 2]) self.gt_classes  torch.Size([128])


    
    def smooth_l1_loss_ANCHOR_REGidual_GT_normal(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """

        if self._no_instances:
            return 0.0 * self.anchor_residual_pred.sum()


        device = self.pred_proposal_deltas.device

        


        bg_class_ind = self.anchor_normal_score.shape[1] - 1
        fg_inds = nonzero_tuple((self.gt_anchor_normal_classes >= 0) & (self.gt_anchor_normal_classes < bg_class_ind))[0]
        # gt_res =  mirror_normal - anchor_normal[i] 
        current_gt_anchor_normal_res =torch.stack([\
                torch.tensor(self.anchor_normals[self.gt_anchor_normal_classes[idx]], device=device) + \
                self.gt_anchor_normal_residuals[idx] - \
                (torch.tensor(self.anchor_normals[self.anchor_normal_score[idx].argmax(dim=0)], device=device) \
                     if self.anchor_normal_score[idx].argmax(dim=0) < bg_class_ind \
                      else self.gt_anchor_normal_residuals[idx] ) \
                for idx in fg_inds], dim=0)


        residual_dim = self.anchor_normals.shape[1]
        fg_gt_classes = self.gt_anchor_normal_classes[fg_inds]
        gt_class_cols = residual_dim * fg_gt_classes[:, None] + torch.arange(residual_dim, device=device)


        loss_ANCHOR_REG_reg = smooth_l1_loss(
            self.anchor_residual_pred[fg_inds[:, None], gt_class_cols],
            current_gt_anchor_normal_res,
            self.smooth_l1_beta,
            reduction="sum",
        )

        loss_ANCHOR_REG_reg = loss_ANCHOR_REG_reg / self.gt_anchor_normal_classes.numel()
        return loss_ANCHOR_REG_reg


    def smooth_l1_loss_ANCHOR_REGidual_GT_residual(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """

        if self._no_instances or not self.ANCHOR_REG:
            return 0.0 * self.anchor_residual_pred.sum()


        device = self.pred_proposal_deltas.device

        bg_class_ind = self.anchor_normal_score.shape[1] - 1
        fg_inds = nonzero_tuple((self.gt_anchor_normal_classes >= 0) & (self.gt_anchor_normal_classes < bg_class_ind))[0]

        residual_dim = 3
        fg_gt_classes = self.gt_anchor_normal_classes[fg_inds]
        gt_class_cols = residual_dim * fg_gt_classes[:, None] + torch.arange(residual_dim, device=device)


        loss_ANCHOR_REG_reg = smooth_l1_loss(
            self.anchor_residual_pred[fg_inds[:, None], gt_class_cols],
            self.gt_anchor_normal_residuals[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )

        loss_ANCHOR_REG_reg = loss_ANCHOR_REG_reg / self.gt_anchor_normal_classes.numel()

        return loss_ANCHOR_REG_reg

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """

        if self.is_training:
            if self.ANCHOR_REG_method == 1:
                return { 
                    "anchor_cls" : self.softmax_cross_entropy_loss_anchor_normal(),
                    "anchor_residual_reg" : self.smooth_l1_loss_ANCHOR_REGidual_GT_residual(),
                    "loss_cls": self.softmax_cross_entropy_loss(),
                    "loss_box_reg": self.smooth_l1_loss()
                    
                }
            elif self.ANCHOR_REG_method == 2:
                return { 
                    "anchor_cls" : self.softmax_cross_entropy_loss_anchor_normal(),
                    "anchor_residual_reg" : self.smooth_l1_loss_ANCHOR_REGidual_GT_normal(),
                    "loss_cls": self.softmax_cross_entropy_loss(),
                    "loss_box_reg": self.smooth_l1_loss()
                }
        else:
            return {
                "loss_cls": None,
                "loss_box_reg": None,
                "anchor_cls" : None,
                "anchor_residual_reg" : None
            }



    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return mirror3d_fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class Mirror3d_FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        anchor_normal_class_num,
        anchor_normals=None,
        ANCHOR_REG_method=1,
        anchor_cls=False,
        ANCHOR_REG=False,
        OBJECT_CLS=True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        self.anchor_normal_score = nn.Linear(input_size, anchor_normal_class_num) #  changed !!! 
        self.anchor_parameter = nn.Linear(input_size, anchor_normal_class_num * 3 ) #  changed !!!

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.anchor_normals = anchor_normals
        self.ANCHOR_REG_method = ANCHOR_REG_method
        self.anchor_cls = anchor_cls
        self.ANCHOR_REG = ANCHOR_REG
        self.OBJECT_CLS = OBJECT_CLS

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "anchor_normal_class_num"   : cfg.ANCHOR_NORMAL_CLASS_NUM + 1, #! + 1 background; background is the last class
            "anchor_normals" : np.load(cfg.ANCHOR_NORMAL_NYP),
            "ANCHOR_REG_method" : cfg.ANCHOR_REG_METHOD,
            "anchor_cls" : cfg.ANCHOR_CLS,
            "ANCHOR_REG" : cfg.ANCHOR_REG,
            "OBJECT_CLS" : cfg.OBJECT_CLS
            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: Nx(K+1) scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        anchor_normal_score = self.anchor_normal_score(x)
        anchor_residual_pred = self.anchor_parameter(x)
        return scores, proposal_deltas , anchor_normal_score , anchor_residual_pred

    def losses(self, predictions, proposals): # changed !!! : add gt *** here (3)
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas ,anchor_normal_score , anchor_residual_pred=predictions
        return Mirror3d_FastRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta,anchor_normal_score , anchor_residual_pred, self.anchor_normals,self.ANCHOR_REG_method , self.anchor_cls, self.ANCHOR_REG, self.training # changed !!! add gt *** here (3.2)
        ).losses()

    def inference(self, predictions, proposals ): 
        """
        Returns:
            list[Instances]: same as `mirror3d_fast_rcnn_inference`.
            list[Tensor]: same as `mirror3d_fast_rcnn_inference`.
        """
        # TODO_ get gt box match here  
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals) 
        anchor_scores = self.predict_anchor_cls(predictions, proposals)
        residual = self.predict_residual(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        #  get top score output changed
        return mirror3d_fast_rcnn_inference( 
            boxes,
            scores,
            anchor_scores,
            residual,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.OBJECT_CLS
        )
    def chris_eval_match_gt(self, scores, boxes):
        
        gt_box = [317.0,101.0,317.0+130.0,101.0+370.0]
        match_id = [0]*scores.shape[0]
        for box_index, one_box in enumerate(boxes):
            IOU = self.get_rec_IOU(one_box, gt_box)
            if IOU > 0.5:
                match_id[box_index] = 1
        print( "matched box count :",sum(match_id), "/ ",len(match_id) )
        return match_id
    
    def get_rec_IOU(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
    
        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
    
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas,_,_ = predictions 
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_residual(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, _,_,residual = predictions 
        num_prop_per_image = [len(p) for p in proposals]
        return residual.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _ ,_,_= predictions 
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


    def predict_anchor_cls(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        _, _ ,scores,_= predictions 
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
        
   