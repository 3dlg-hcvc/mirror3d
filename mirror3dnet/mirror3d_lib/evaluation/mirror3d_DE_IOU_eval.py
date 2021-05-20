#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import torch
from utils.mirror3d_metrics import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *

from detectron2.data import (
    MetadataCatalog,
)

import logging
import time
import numpy as np
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.events import get_event_storage

class Mirror3DNet_Eval:

    def __init__(self, output_list, cfg):
        self.output_list = output_list
        self.cfg = cfg
        self.mean_IOU = 0
        print("########## cfg.SEED ##########", cfg.SEED)
        log_file_save_path = os.path.join(self.cfg.OUTPUT_DIR, "eval_result.log")
        logging.basicConfig(filename=log_file_save_path, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)
        self.logger = logging

        if self.cfg.TRAIN_COCO_JSON.find("nyu") > 0:
            self.dataset_name = "nyu"
        elif self.cfg.TRAIN_COCO_JSON.find("m3d") > 0:
            self.dataset_name = "m3d"
        elif self.cfg.TRAIN_COCO_JSON.find("scannet") > 0:
            self.dataset_name = "scannet"

    def eval_main(self):

        if self.cfg.EVAL_SAVE_MASKED_IMG:
            self.save_masked_image(self.output_list)

        if self.cfg.EVAL_MASK_IOU:
            print("calculate IOU, f_measure, MAE ..")
            self.eval_seg(self.output_list)

        # ----------- evaluate \mnet/ PlaneRCNN model's predicted depth + Mirror3d -----------
        if self.cfg.EVAL_BRANCH_REF_DEPTH:
            print("eval DE_pred + refine ...")
            self.refine_DEbranch_predD_and_eval(self.output_list)

        if self.cfg.EVAL_BRANCH_ORI_DEPTH:
            print("eval DE_pred ...")
            self.eval_raw_DEbranch_predD(self.output_list)
        # ----------- evaluate REF_DEPTH_TO_REFINE (coco.json) + Mirror3d -----------
        if "raw" in self.cfg.REF_MODE:
            print("eval rawD + refine ...")
            self.refine_raw_inputD_and_eval(self.output_list)
        
        # ----------- evaluate REF_DEPTH_TO_REFINE (output from init_depth_generator/ \mnet'DE branch) + Mirror3d -----------
        if self.cfg.EVAL_INPUT_REF_DEPTH and "raw" not in self.cfg.REF_MODE:
            print("eval input {} + refine ...".format(self.cfg.REF_DEPTH_TO_REFINE))
            self.refine_input_txtD_and_eval(self.output_list)
    
        




    def refine_input_txtD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if self.cfg.REF_DEPTH_TO_REFINE.find("SAIC") > 0:
            Input_tag = "RGBD"
            method_tag = "saic + \mnet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("BTS") > 0:
            Input_tag = "RGB"
            method_tag = "BTS + \mnet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("VNL") > 0:
            Input_tag = "RGB"
            method_tag = "VNL + \mnet"
        elif not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "PlaneRCNN"
        else:
            Input_tag = "RGB"
            method_tag = "\mnet"
        
        train_with_refD=None 
        if self.cfg.REF_DEPTH_TO_REFINE.find("ref") > 0:
            train_with_refD = True
        else:
            train_with_refD = False

        mirror3d_eval = Mirror3d_eval(train_with_refD=train_with_refD, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        input_txt = read_txt(self.cfg.REF_DEPTH_TO_REFINE)

        imgPath_preDPath = dict()
        for line in input_txt:
            img_paths = line.strip().split()
            imgPath_preDPath[img_paths[0]] = img_paths[-1]

        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            color_img_path = one_input[0]["img_path"]

            pred_mask = np.zeros(instances.image_size)
            pred_mask = pred_mask.astype(bool)

            other_predD_path = imgPath_preDPath[color_img_path]
            depth_to_ref = cv2.imread(other_predD_path, cv2.IMREAD_ANYDEPTH)

            if instances.to("cpu").has("pred_masks"):
                for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
                    
                    to_refine_area = one_pred_mask.numpy().astype(bool)
                    to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
                    if to_refine_area.sum() == 0:
                        continue
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
                        continue
                    
                    pred_normal = anchor_normal[instances.to("cpu").pred_anchor_classes[index]] +  instances.to("cpu").pred_residuals[index].numpy()
                    pred_normal = unit_vector(pred_normal)

                    if "border" in self.cfg.REF_MODE :
                        depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, depth_to_ref)
                    else:
                        depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, depth_to_ref)
            depth_to_ref[depth_to_ref<0] = 0
            mirror3d_eval.compute_and_update_mirror3D_metrics(depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
            if self.cfg.EVAL_SAVE_DEPTH:
                refined_input_txt_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "refined_input_txt_pred_depth")
                os.makedirs(refined_input_txt_output_folder, exist_ok=True)
                mirror3d_eval.save_result(refined_input_txt_output_folder, depth_to_ref/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, color_img_path)

        print("############# Result of 'txt {} + Mirror3dNet' #############".format(self.cfg.REF_DEPTH_TO_REFINE))
        if self.cfg.EVAL_SAVE_DEPTH:
            print("##### result saved to ##### {}".format(os.path.join(self.cfg.OUTPUT_DIR,"color_mask_gtD_predD.txt")))
        mirror3d_eval.print_mirror3D_score()

    def refine_raw_inputD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        mirror3d_eval_sensorD = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="sensor-D", method_tag="*",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_meshD = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="mesh-D", method_tag="*",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_hole = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="sensor-D", method_tag="\mnet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_mesh = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="mesh-D", method_tag="\mnet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        have_mesh_D = False
        imgPath_info = dict()

        input_json = read_json(self.cfg.VAL_COCO_JSON)
        for item in input_json["images"]:
            img_path = os.path.join(self.cfg.VAL_IMG_ROOT, item["img_path"])
            
            imgPath_info[img_path] = item
        
        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            color_img_path = one_input[0]["img_path"]
            hole_raw_depth_path = os.path.join(self.cfg.VAL_IMG_ROOT, imgPath_info[color_img_path]["hole_raw_path"])
            mesh_raw_depth_path = os.path.join(self.cfg.VAL_IMG_ROOT, imgPath_info[color_img_path]["mesh_raw_path"])

            pred_mask = np.zeros(instances.image_size)
            pred_mask = pred_mask.astype(bool)

            hole_depth_to_ref = cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)
            if mesh_raw_depth_path != hole_raw_depth_path:
                mesh_depth_to_ref = cv2.imread(mesh_raw_depth_path, cv2.IMREAD_ANYDEPTH)

            hole_depth_to_ref_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "hole_raw_depth_mirror3d_refine")
            os.makedirs(hole_depth_to_ref_output_folder, exist_ok=True)
            mesh_depth_to_ref_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "mesh_raw_depth_mirror3d_refine")
            os.makedirs(mesh_depth_to_ref_output_folder, exist_ok=True)


            if instances.to("cpu").has("pred_masks"):
                for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
                    
                    to_refine_area = one_pred_mask.numpy().astype(bool)
                    to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
                    if to_refine_area.sum() == 0:
                        continue
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
                        continue
                    pred_normal = anchor_normal[instances.to("cpu").pred_anchor_classes[index]] +  instances.to("cpu").pred_residuals[index].numpy()
                    pred_normal = unit_vector(pred_normal)

                    if mesh_raw_depth_path != hole_raw_depth_path:
                        if "border" in self.cfg.REF_MODE :
                            mesh_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, mesh_depth_to_ref)
                        else:
                            mesh_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, mesh_depth_to_ref)

                    if "border" in self.cfg.REF_MODE :
                        hole_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, hole_depth_to_ref)
                    else:
                        hole_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, hole_depth_to_ref)
            
            hole_depth_to_ref[hole_depth_to_ref<0] = 0
            
            mirror3d_eval_hole.compute_and_update_mirror3D_metrics(hole_depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
            mirror3d_eval_sensorD.compute_and_update_mirror3D_metrics(cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
            if self.cfg.EVAL_SAVE_DEPTH:
                mirror3d_eval_hole.save_result(hole_depth_to_ref_output_folder, hole_depth_to_ref/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, color_img_path)

            if mesh_raw_depth_path != hole_raw_depth_path:
                have_mesh_D = True
                mesh_depth_to_ref[mesh_depth_to_ref<0] = 0
                mirror3d_eval_mesh.compute_and_update_mirror3D_metrics(mesh_depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
                mirror3d_eval_meshD.compute_and_update_mirror3D_metrics(cv2.imread(mesh_raw_depth_path, cv2.IMREAD_ANYDEPTH)/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
                if self.cfg.EVAL_SAVE_DEPTH:
                    mirror3d_eval_mesh.save_result(mesh_depth_to_ref_output_folder, mesh_depth_to_ref/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, color_img_path)
            else:
                os.rmdir(mesh_depth_to_ref_output_folder)
        
        print("############# hole raw depth + Mirror3dNet result #############")
        mirror3d_eval_hole.print_mirror3D_score()
        print("############# hole raw (sensor) depth result #############")
        mirror3d_eval_sensorD.print_mirror3D_score()
        if have_mesh_D:
            print("############# mesh raw depth + Mirror3dNet result #############")
            mirror3d_eval_mesh.print_mirror3D_score()
            print("############# mesh raw (meshD) depth refine result #############")
            mirror3d_eval_meshD.print_mirror3D_score()

    def eval_raw_DEbranch_predD(self, output_list):

        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        if not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "PlaneRCNN-DE"
        else:
            Input_tag = "RGB"
            method_tag = "\mnet-DE"
        
        if self.cfg.REFINED_DEPTH:
            train_with_refD = True
        else:
            train_with_refD = False

        mirror3d_eval = Mirror3d_eval(train_with_refD=train_with_refD, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            gt_depth = cv2.imread(one_input[0]["mesh_refined_path"], cv2.IMREAD_ANYDEPTH)

            np_pred_depth = pred_depth.astype(np.uint16)

            mirror3d_eval.compute_and_update_mirror3D_metrics(np_pred_depth/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])

            if self.cfg.EVAL_SAVE_DEPTH:
                raw_branch_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "DE_branch_pred_depth")
                os.makedirs(raw_branch_output_folder, exist_ok=True)
                mirror3d_eval.save_result(raw_branch_output_folder, np_pred_depth/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])
            
        print("evaluate DE result for {}".format(method_tag))
        self.logger.info("evaluate DE result for {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()

    def refine_DEbranch_predD_and_eval(self, output_list):

        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "PlaneRCNN"
        else:
            Input_tag = "RGB"
            method_tag = "\mnet"

        train_with_refD=None
        if self.cfg.MODEL.WEIGHTS.find("ref") > 0:
            train_with_refD = True
        else:
            train_with_refD = False
        mirror3d_eval = Mirror3d_eval(train_with_refD=train_with_refD, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            np_pred_depth = pred_depth.copy()
            depth_p = pred_depth.copy()
            
            
            # -------------- refine depth with predict anchor normal ------------
            instances = one_output[0][0]["instances"]
            anchor_normals = np.load(self.cfg.ANCHOR_NORMAL_NYP)
            
            for instance_idx, pred_anchor_normal_class in enumerate(instances.pred_anchor_classes):
                instance_mask = instances.pred_masks[instance_idx].detach().cpu().numpy()
                
                if pred_anchor_normal_class >= anchor_normals.shape[0]:
                    continue
                else:
                    if self.cfg.ANCHOR_REG:
                        plane_normal = anchor_normals[pred_anchor_normal_class] + instances.pred_residuals[instance_idx].detach().cpu().numpy()
                    else:
                        plane_normal = anchor_normals[pred_anchor_normal_class]
                a, b, c = unit_vector(plane_normal)
                if "border" in self.cfg.REF_MODE:
                    depth_p = refine_depth_fun.refine_depth_by_mirror_border(instance_mask, [a, b, c], pred_depth)
                else:
                    depth_p = refine_depth_fun.refine_depth_by_mirror_area(instance_mask, [a, b, c], pred_depth)

            np_pred_depth = np_pred_depth.astype(np.uint16)
            depth_p = depth_p.astype(np.uint16)
            mirror3d_eval.compute_and_update_mirror3D_metrics(depth_p/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])

            if self.cfg.EVAL_SAVE_DEPTH:
                refined_DE_branch_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "refined_DE_branch_pred_depth")
                os.makedirs(refined_DE_branch_output_folder, exist_ok=True)
                mirror3d_eval.save_result(refined_DE_branch_output_folder, depth_p/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])
            
        print("eval refined depth from DE branch : {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()

    def save_masked_image(self, output_list):
        masked_img_save_folder = os.path.join(self.cfg.OUTPUT_DIR, "masked_img")
        os.makedirs(masked_img_save_folder, exist_ok=True)

        for one_output, one_input in output_list:
            
            instances = one_output[0][0]["instances"]
            img_path = one_input[0]["img_path"]
            if instances.pred_boxes.tensor.shape[0] <= 0:
                print("######## no detection :", img_path)
                continue
            img = cv2.imread(img_path)
            v = Visualizer(img[:, :, ::-1], 
                metadata=MetadataCatalog.get("test_10_normal_mirror"), 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW  
                )
            v = v.draw_instance_predictions(instances.to("cpu")) 
            output_img = v.get_image()[:, :, ::-1]
            img_save_path = os.path.join(masked_img_save_folder, img_path.split("/")[-1])
            cv2.imwrite(img_save_path, output_img)
            print("masked image saved to : ", img_save_path)
            

    def eval_seg(self, output_list):

        eval_seg_fun = Mirror_seg_eval(self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            mask_path = one_input[0]["img_path"].replace("raw","instance_mask")
            if not os.path.exists(mask_path) or "no_mirror" in one_input[0]["img_path"]:
                continue
            GT_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            GT_mask = GT_mask > 0
            pred_mask = np.zeros_like(GT_mask)
            pred_mask = pred_mask.astype(bool)
            if instances.to("cpu").has("pred_masks"):
                for one_pred_mask in instances.to("cpu").pred_masks:
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    pred_mask = pred_mask.numpy().astype(bool)
                
            eval_seg_fun.compute_and_update_seg_metrics(pred_mask, GT_mask)

        eval_seg_fun.print_seg_score()
        IOU_list, f_measure_list, MAE_list = eval_seg_fun.get_results()
        self.mean_IOU = np.mean(IOU_list)
        if not self.cfg.EVAL:
            storage = get_event_storage()
            storage.put_scalar("mean IOU",np.mean(IOU_list))
            storage.put_scalar("mean f measure",np.mean(f_measure_list))
            storage.put_scalar("mean MAE",np.mean(MAE_list))
