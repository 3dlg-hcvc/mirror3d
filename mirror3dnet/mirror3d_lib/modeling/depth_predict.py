import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

import cv2

class Depth(nn.Module):
    def __init__(self, num_output_channels=1):
        super(Depth, self).__init__()
        self.num_output_channels = num_output_channels        
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,128),
            # nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32,64),
            # nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.depth_pred = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

        self.crop = True
        return
    
    def forward(self, feature_maps, gt_depths, istraining):


        x = self.deconv1(self.conv1(feature_maps["p6"]))
        if x.shape[2] != feature_maps["p5"].shape[2]:
            x = x[:, :, :feature_maps["p5"].shape[2] ,:feature_maps["p5"].shape[3]]
        x = self.deconv2(torch.cat([self.conv2(feature_maps["p5"]), x], dim=1))
        x = self.deconv3(torch.cat([self.conv3(feature_maps["p4"]), x], dim=1))
        x = self.deconv4(torch.cat([self.conv4(feature_maps["p3"]), x], dim=1))
        x = self.deconv5(torch.cat([self.conv5(feature_maps["p2"]), x], dim=1))
        x = self.depth_pred(x) #  torch.Size([1, 1, 240, 320])

        x = torch.nn.functional.interpolate(x, size=gt_depths[0].shape, mode='bilinear')

        depth_loss = sum([torch.sum(torch.abs(x[idx].squeeze() - gt_depths[idx]) ) / torch.clamp(((gt_depths[idx] > 1e-4).float()).sum(), min=1) for idx , one_gt in enumerate(gt_depths)]) / len(gt_depths)
        if torch.isnan(depth_loss):
            print("NaN")
        
        if istraining:
            return {"depth_estimate_loss": depth_loss}
        else:
            return [item.squeeze() for item in x]



def calcXYZModule(config, camera, detections, masks, depth_np, return_individual=False, debug_type=0):
    """Compute a global coordinate map from plane detections"""
    ranges = config.getRanges(camera)
    ranges_ori = ranges
    zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()        
    ranges = torch.cat([zeros, ranges.transpose(1, 2).transpose(0, 1), zeros], dim=1)
    XYZ_np = ranges * depth_np

    if len(detections) == 0:
        detection_mask = torch.zeros((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
        if return_individual:
            return XYZ_np, detection_mask, []
        else:
            return XYZ_np, detection_mask
        pass
    
    plane_parameters = detections[:, 6:9]
    
    XYZ = torch.ones((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda() * 10
    depthMask = torch.zeros((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
    planeXYZ = planeXYZModule(ranges_ori, plane_parameters, width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)
    planeXYZ = planeXYZ.transpose(2, 3).transpose(1, 2).transpose(0, 1)
    zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM, int(planeXYZ.shape[-1])).cuda()
    planeXYZ = torch.cat([zeros, planeXYZ, zeros], dim=1)

    one_hot = True    
    if one_hot:
        for detectionIndex in range(len(detections)):
            mask = masks[detectionIndex]
            with torch.no_grad():
                mask_binary = torch.round(mask)
                pass
            if config.FITTING_TYPE >= 2:
                if (torch.norm(planeXYZ[:, :, :, detectionIndex] - XYZ_np, dim=0) * mask_binary).sum() / torch.clamp(mask_binary.sum(), min=1e-4) > 0.5:
                    mask_binary = torch.zeros(mask_binary.shape).cuda()
                    pass
                pass
            mask_binary = mask_binary * (planeXYZ[1, :, :, detectionIndex] < XYZ[1]).float()
            XYZ = planeXYZ[:, :, :, detectionIndex] * mask_binary + XYZ * (1 - mask_binary)
            depthMask = torch.max(depthMask, mask)
            continue
        XYZ = XYZ * torch.round(depthMask) + XYZ_np * (1 - torch.round(depthMask))
    else:
        background_mask = torch.clamp(1 - masks.sum(0, keepdim=True), min=0)
        all_masks = torch.cat([background_mask, masks], dim=0)
        all_XYZ = torch.cat([XYZ_np.unsqueeze(-1), planeXYZ], dim=-1)
        XYZ = (all_XYZ.transpose(2, 3).transpose(1, 2) * all_masks).sum(1)
        depthMask = torch.ones(depthMask.shape).cuda()
        pass

    if debug_type == 2:
        XYZ = XYZ_np
        pass

    if return_individual:
        return XYZ, depthMask, planeXYZ.transpose(2, 3).transpose(1, 2).transpose(0, 1)
    return XYZ, depthMask



def planeXYZModule(ranges, planes, width, height, max_depth=10):
    """Compute plane XYZ from plane parameters
    ranges: K^(-1)x
    planes: plane parameters
    
    Returns:
    plane depthmaps
    """
    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)
    normalXYZ = torch.matmul(ranges, planeNormals.transpose(0, 1))
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
    return planeDepths.unsqueeze(-1) * ranges.unsqueeze(2)
