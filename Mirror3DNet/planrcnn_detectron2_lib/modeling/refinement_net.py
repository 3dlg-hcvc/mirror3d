"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

import numpy as np
import os

from models.modules import *
from utils import *

class PlaneToDepth(torch.nn.Module):
    def __init__(self, normalized_K = True, normalized_flow = True, inverse_depth = True, W = 64, H = 48):

        super(PlaneToDepth, self).__init__()

        self.normalized_K = normalized_K
        self.normalized_flow = normalized_flow
        self.inverse_depth = inverse_depth

        with torch.no_grad():
            self.URANGE = ((torch.arange(W).float() + 0.5) / W).cuda().view((1, -1)).repeat(H, 1)
            self.VRANGE = ((torch.arange(H).float() + 0.5) / H).cuda().view((-1, 1)).repeat(1, W)
            self.ONES = torch.ones((H, W)).cuda()
            pass
        
    def forward(self, intrinsics, plane, return_XYZ=False):

        """
        :param K1: intrinsics of 1st image, 3x3
        :param K2: intrinsics of 2nd image, 3x3
        :param depth: depth map of first image, 1 x height x width
        :param rot: rotation from first to second image, 3
        :param trans: translation from first to second, 3
        :return: normalized flow from 1st image to 2nd image, 2 x height x width
        """

        with torch.no_grad():
            urange = (self.URANGE * intrinsics[4] - intrinsics[2]) / intrinsics[0]
            vrange = (self.VRANGE * intrinsics[5] - intrinsics[3]) / intrinsics[1]
            ranges = torch.stack([urange,
                                  self.ONES,
                                  -vrange], -1)
            pass
        
        # chris : get offset of the plane
        planeOffset = torch.norm(plane, dim=-1)
        # chris : get the normal of the plane
        planeNormal = plane / torch.clamp(planeOffset.unsqueeze(-1), min=1e-4)
        # chris : get the depth map of the plane
        depth = planeOffset / torch.clamp(torch.sum(ranges.unsqueeze(-2) * planeNormal, dim=-1), min=1e-4)
        depth = torch.clamp(depth, min=0, max=10)

        # chris : self.inverse_depth: = True
        if self.inverse_depth:
            # chris : get the inverse of the depth
            depth = invertDepth(depth)
        depth = depth.transpose(1, 2).transpose(0, 1)

        # chris ： return_XYZ = True
        if return_XYZ:
            # return depth of plane region & plane_XYZ in camera cooridinate
            return depth, depth.unsqueeze(-1) * ranges
        return depth        

   
class RefinementBlockMask(torch.nn.Module):
   def __init__(self, options):
       super(RefinementBlockMask, self).__init__()
       self.options = options
       use_bn = False
       self.conv_0 = ConvBlock(3 + 5 + 2, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)       
       self.conv_1_1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
       self.conv_2_1 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

       self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
       self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
       self.pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                 torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

       self.global_up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
       self.global_up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
       self.global_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                       torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))
       self.depth_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                       torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))       
       return

   def accumulate(self, x):
       return torch.cat([x, (x.sum(0, keepdim=True) - x) / max(len(x) - 1, 1)], dim=1)

   # chris : image_1.repeat((len(masks), 1, 1, 1)), prev_predictions, prev_result['plane']
   # chris : prev_parameters is the plane parameter
   # chris : mask = prev_predictions = plane_depth , depth , mask
   def forward(self, image, masks, prev_parameters=None):
       x_mask = masks
       
       x_0 = torch.cat([image, x_mask], dim=1)

       x_0 = self.conv_0(x_0)
       x_1 = self.conv_1(self.accumulate(x_0))
       x_1 = self.conv_1_1(self.accumulate(x_1))
       x_2 = self.conv_2(self.accumulate(x_1))
       x_2 = self.conv_2_1(self.accumulate(x_2))
       
       y_2 = self.up_2(x_2)
       y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
       y_0 = self.pred(torch.cat([y_1, x_0], dim=1))
       
       global_y_2 = self.global_up_2(x_2.mean(dim=0, keepdim=True))
       global_y_1 = self.global_up_1(torch.cat([global_y_2, x_1.mean(dim=0, keepdim=True)], dim=1))
       global_mask = self.global_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1))
       # chris : predict new depth
       depth = self.depth_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1)) + x_mask[:1, :1]

       parameters = prev_parameters
       
       y_0 = torch.cat([global_mask[:, 0], y_0.squeeze(1)], dim=0)
       # return new masks, new depth, previous parameters
       return y_0, depth, parameters


class RefinementNet(nn.Module):

    def __init__(self, options):
        super(RefinementNet, self).__init__()
        self.options = options
        self.refinement_block = RefinementBlockMask(options)


        self.upsample = torch.nn.Upsample(size=(192, 256), mode='bilinear')            
        self.plane_to_depth = PlaneToDepth(normalized_K=True, W=256, H=192)

        return
    
    def forward(self, image_1, camera, prev_result):
        # chris : prev_result = {'mask': masks, 'plane': planes, 'depth': depth_np.unsqueeze(1), 'plane_depth': depth_np.unsqueeze(1)}
        masks = prev_result['mask']
        prev_predictions = torch.cat([torch.cat([prev_result['plane_depth'], prev_result['depth']], dim=1).repeat((len(masks), 1, 1, 1)), masks, (masks.sum(0, keepdim=True) - masks)[:, :1]], dim=1)

        # chris : masks & depth are updated
        # chris : plane = prev_result['plane']
        masks, depth, plane = self.refinement_block(image_1.repeat((len(masks), 1, 1, 1)), prev_predictions, prev_result['plane'])
        result = {}

        result = {'plane': plane, 'depth': depth}

        # chris : get plane depth map from predicted plane
        plane_depths, plane_XYZ = self.plane_to_depth(camera[0], result['plane'], return_XYZ=True)
        all_depths = torch.cat([result['depth'].squeeze(1), plane_depths], dim=0)
        
        all_masks = torch.softmax(masks, dim=0)
        plane_depth = (all_depths * all_masks).sum(0, keepdim=True)

        result['mask'] = masks.unsqueeze(1)
        return result


class RefineModel(nn.Module):
    def __init__(self, options):
        super(RefineModel, self).__init__()

        self.options = options
        
        K = [[0.89115971,  0,  0.5],
             [0,  1.18821287,  0.5],
             [0,           0,    1]]
        with torch.no_grad():
            self.intrinsics = torch.Tensor(K).cuda()
            pass
        """ the whole network """

        self.refinement_net = RefinementNet(options)
        W, H = 64, 48
        self.upsample = torch.nn.Upsample(size=(H, W), mode='bilinear')

        if 'crfrnn_only' in self.options.suffix:
            self.plane_to_depth = PlaneToDepth(normalized_K = True, W=256, H=192)
            self.crfrnn = CRFRNNModule(image_dims=(192, 256), num_iterations=5)
            pass
        return

    
    def forward(self, image, camera, masks, planes, plane_depth, depth_np, gt_dict={}):

        x = {'mask': masks, 'plane': planes, 'depth': depth_np.unsqueeze(1), 'plane_depth': depth_np.unsqueeze(1)}
        result = self.refinement_net(image, camera, x)
        
      
        return result
