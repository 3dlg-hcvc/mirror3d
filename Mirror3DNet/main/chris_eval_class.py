#!/usr/bin/python
# -*- coding: UTF-8 -*-



import os
import sys
import torch
from utils.Mirror3D_eval import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *

from planrcnn_detectron2_lib.data import (
    MetadataCatalog,
)

import logging
import os
import time
import cv2
from planrcnn_detectron2_lib.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import numpy as np
from detectron2.utils.events import get_event_storage


class Chris_eval:

    def __init__(self, output_list, cfg):
        self.output_list = output_list
        self.cfg = cfg
        log_file_save_path = os.path.join(self.cfg.OUTPUT_DIR, "eval_result.log")
        logging.basicConfig(filename=log_file_save_path, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)
        self.logger = logging
    def eval_main(self):

        # ----------- evaluate Mirror3DNet/ planercnn model's predicted depth + Mirror3d -----------
        if self.cfg.EVAL_BRANCH_REF_DEPTH:
            self.refine_DEbranch_predD_and_eval(self.output_list)
            # self.eval_plane_refine_depth(self.time_tag, self.output_list)

        if self.cfg.EVAL_BRANCH_ORI_DEPTH:
            self.eval_raw_DEbranch_predD(self.output_list)

        # ----------- evaluate REF_DEPTH_TO_REFINE (coco.json) + Mirror3d -----------
        if self.cfg.EVAL_INPUT_REF_DEPTH and "raw" in self.cfg.REF_MODE:
            self.refine_raw_inputD_and_eval(self.output_list)
        
        # ----------- evaluate REF_DEPTH_TO_REFINE (output from init_depth_generator/ Mirror3DNet'DE branch) + Mirror3d -----------
        if self.cfg.EVAL_INPUT_REF_DEPTH and "raw" not in self.cfg.REF_MODE:
            self.refine_input_txtD_and_eval(self.output_list)
            # self.raw_D_refined(self.time_tag, self.output_list)
    
        if self.cfg.EVAL_MASK_IOU:
            self.eval_seg(self.output_list)

        if self.cfg.EVAL_SAVE_MASKED_IMG:
            save_masked_image(self.output_list)

    # def planercnn_DE(self, time_tag, output_list):

    #     output_folder = "/project/3dlg-hcvc/jiaqit/waste/PE_result_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    #     os.makedirs(output_folder, exist_ok=True)
    #     output_info_txt = os.path.join(output_folder, "PE_gt_pred_color_mask.txt") # gt_pred_color_mask_list

    #     rmse_list = np.zeros(len(output_list), np.float32)
    #     s_rmse_list = np.zeros(len(output_list), np.float32)

    #     for i, one_output in enumerate(output_list):
    #         pred_depth = one_output[0][1][0].detach().cpu().numpy()
    #         gt_depth_path = one_output[1][0]["mesh_refined_path"]
    #         gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH)


    #         img_save_name = gt_depth_path.split("/")[-1]
    #         img_save_path = os.path.join(output_folder, img_save_name)

    #         cv2.imwrite(img_save_path, np.array(pred_depth, dtype=np.uint16))
    #         mask_path = one_output[1][0]["img_path"].replace("raw", "instance_mask")
    #         with open(output_info_txt, "a") as file:
    #             if os.path.exists(mask_path):
    #                 file.write("{} {} {} {}".format(gt_depth_path, img_save_path, one_output[1][0]["img_path"],mask_path ))
    #             else:
    #                 file.write("{} {} {} {}".format(gt_depth_path, img_save_path, one_output[1][0]["img_path"],"None" ))
    #             file.write("\n")

    #     print("################## result saved to ##################", output_info_txt)

    # def eval_only_DE(self, time_tag, output_list):
    #     def compute_errors(gt, pred, mask, depth_shift): # gt and pred are in m

    #         gt = gt/float(depth_shift)
    #         pred = pred/float(depth_shift)
            
    #         min_depth_eval = 1e-3
    #         max_depth_eval = 10

    #         pred[pred < min_depth_eval] = min_depth_eval
    #         # pred[pred > max_depth_eval] = max_depth_eval
    #         pred[np.isinf(pred)] = max_depth_eval

    #         gt[np.isinf(gt)] = 0
    #         gt[np.isnan(gt)] = 0

    #         valid_mask = gt > min_depth_eval #  np.logical_and(gt > min_depth_eval)#, gt < max_depth_eval
    #         scale = np.sum(pred[valid_mask]*gt[valid_mask])/np.sum(pred[valid_mask]**2)
    #         valid_mask = np.logical_and(valid_mask, mask)

    #         gt = gt[valid_mask]
    #         pred = pred[valid_mask]

            
    #         rmse = (gt - pred) ** 2
    #         rmse = np.sqrt(rmse.mean())

    #         scaled_rms = np.sqrt(((scale * pred-gt)**2).mean())


    #         return rmse, scaled_rms

    #     rmse_list = np.zeros(len(output_list), np.float32)
    #     s_rmse_list = np.zeros(len(output_list), np.float32)

    #     for i, one_output in enumerate(output_list):
    #         pred_depth = one_output[0][1][0].detach().cpu().numpy()
    #         gt_depth = cv2.imread(one_output[1][0]["mesh_refined_path"], cv2.IMREAD_ANYDEPTH)
    #         one_result = compute_errors(gt_depth, pred_depth, True, self.cfg.DEPTH_SHIFT)
    #         rmse_list[i], s_rmse_list[i] = one_result

    #     storage = get_event_storage()
    #     storage.put_scalar("DE whole_img RMSE",rmse_list.mean())
    #     storage.put_scalar("DE whole_img s-RMSE",s_rmse_list.mean())
    #     print("######################### DE whole_img RMSE {}".format(rmse_list.mean()))
    #     print("######################### DE whole_img s-RMSE {}".format(s_rmse_list.mean()))

    def refine_input_txtD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if self.cfg.REF_DEPTH_TO_REFINE.find("saic") > 0:
            Input_tag = "RGBD"
            method_tag = "saic+M3DNet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("bts") > 0:
            Input_tag = "RGB"
            method_tag = "bts+M3DNet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("vnl") > 0:
            Input_tag = "RGB"
            method_tag = "vnl+M3DNet"
        elif not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "planercnn"
        else:
            Input_tag = "RGB"
            method_tag = "Mirror3DNet"

        mirror3d_eval = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT)

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
                mirror3d_eval.save_result(self.cfg.OUTPUT_DIR, depth_to_ref, self.cfg.DEPTH_SHIFT, color_img_path)

        print("############# Result of 'txt {} + Mirror3dNet' #############".format(self.cfg.REF_DEPTH_TO_REFINE))
        if self.cfg.EVAL_SAVE_DEPTH:
            print("##### result saved to ##### {}".format(os.path.join(self.cfg.OUTPUT_DIR,"color_mask_gtD_predD.txt")))
        mirror3d_eval.print_mirror3D_score()

    def refine_raw_inputD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        mirror3d_eval_hole = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="RGBD", method_tag="Mirror3DNet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT)
        mirror3d_eval_mesh = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag="RGBD", method_tag="Mirror3DNet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT)

        imgPath_info = dict()

        input_json = read_json(self.cfg.REF_DEPTH_TO_REFINE)
        for item in input_json["images"]:
            imgPath_info[item["img_path"]] = item
        
        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            color_img_path = one_input[0]["img_path"]
            hole_raw_depth_path = imgPath_info[color_img_path]["hole_raw_path"]
            mesh_raw_depth_path = imgPath_info[color_img_path]["mesh_raw_path"]

            pred_mask = np.zeros(instances.image_size)
            pred_mask = pred_mask.astype(bool)

            hole_depth_to_ref = cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)
            if mesh_raw_depth_path != hole_raw_depth_path:
                mesh_depth_to_ref = cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)

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
            if self.cfg.EVAL_SAVE_DEPTH:
                mirror3d_eval_hole.save_result(hole_depth_to_ref_output_folder, hole_depth_to_ref, args.DEPTH_SHIFT, color_img_path)

            if mesh_raw_depth_path != hole_raw_depth_path:
                mesh_depth_to_ref[mesh_depth_to_ref<0] = 0
                mirror3d_eval_mesh.compute_and_update_mirror3D_metrics(mesh_depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path)
                if self.cfg.EVAL_SAVE_DEPTH:
                    mirror3d_eval_mesh.save_result(mesh_depth_to_ref_output_folder, mesh_depth_to_ref, args.DEPTH_SHIFT, color_img_path)
            else:
                os.rmdir(mesh_depth_to_ref_output_folder)
        
        print("############# hole raw depth refine result #############")
        mirror3d_eval_hole.print_mirror3D_score()
        print("############# mesh raw depth refine result #############")
        mirror3d_eval_mesh.print_mirror3D_score()

    def eval_raw_DEbranch_predD(self, output_list):

        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        if not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "planercnn"
        else:
            Input_tag = "RGB"
            method_tag = "Mirror3DNet"

        mirror3d_eval = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            gt_depth = cv2.imread(one_input[0]["mesh_refined_path"], cv2.IMREAD_ANYDEPTH)

            np_pred_depth = pred_depth.astype(np.uint16)

            mirror3d_eval.compute_and_update_mirror3D_metrics(np_pred_depth/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])

            if self.cfg.EVAL_SAVE_DEPTH:
                mirror3d_eval.save_result(self.cfg.OUTPUT_DIR, np_pred_depth, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])
            
        print("evaluate DE result for {}".format(method_tag))
        self.logger.info("evaluate DE result for {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()

    def refine_DEbranch_predD_and_eval(self, output_list):

        refine_depth_fun = Refine_depth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if not self.cfg.OBJECT_CLS:
            Input_tag = "RGB"
            method_tag = "planercnn"
        else:
            Input_tag = "RGB"
            method_tag = "Mirror3DNet"

        mirror3d_eval = Mirror3d_eval(train_with_refD=None, logger=self.logger,Input_tag=Input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            np_pred_depth = pred_depth.copy()
            
            
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
                mirror3d_eval.save_result(self.cfg.OUTPUT_DIR, depth_p, self.cfg.DEPTH_SHIFT, one_input[0]["img_path"])
            
        print("eval refined depth from DE branch : {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()


    # def eval_plane_refine_depth(self, time_tag, output_list):


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

    #     if self.cfg.EVAL_SAVE_DEPTH:
    #         output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag)
    #         os.makedirs(output_folder, exist_ok=True)
    #         info_txt_save_path = os.path.join(output_folder, "mask_gtDepth_pDepth_npDepth.txt")
    #         AR_correct_info_txt_save_path = os.path.join(output_folder, "AR_correct_mask_gtDepth_pDepth_npDepth.txt")
    #         if os.path.exists(info_txt_save_path):
    #             os.system('rm ' + info_txt_save_path)
    #         if os.path.exists(AR_correct_info_txt_save_path):
    #             os.system('rm ' + AR_correct_info_txt_save_path)
        

    #     if self.cfg.DEPTH_EST:
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
    #             gt_depth = cv2.imread(one_input[0]["mesh_refined_path"], cv2.IMREAD_ANYDEPTH)
    #             pred_depth = one_output[1][0].detach().cpu().numpy()
    #             pred_depth[pred_depth<0] = 0

    #             silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i], scaled_rms[i] = compute_errors(gt_depth, pred_depth, True)
    #             if self.cfg.EVAL_SAVE_DEPTH:
    #                 current_test_root = ""
    #                 raw_input_img_path = one_input[0]["file_name"] 
    #                 for one_test_img_root in self.cfg.TEST_IMG_ROOT:
    #                     if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
    #                         current_test_root = one_test_img_root
    #                         break
    #                 gt_depth_path = one_input[0]["mesh_refined_path"]
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
    #         print("self.cfg.OUTPUT_DIR : ",self.cfg.OUTPUT_DIR)
    #         print("self.cfg.DEPTH_SHIFT : ", self.cfg.DEPTH_SHIFT)
    #         print("--------------------{:20}--------------------".format("whole image"))
    #         print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
    #             'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10', "scaled_rms"))
    #         print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
    #             d1.mean(), d2.mean(), d3.mean(),
    #         abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean(), scaled_rms.mean()))


    #         if not self.cfg.EVAL:
    #             storage = get_event_storage()
    #             storage.put_scalar("DE whole_img scaled_rms",np.mean(scaled_rms.mean()))
    #             storage.put_scalar("DE whole_img rms",np.mean(rms.mean()))
    #             return



    #     for one_output, one_input in output_list:
            
    #         pred_depth = one_output[1][0].detach().cpu().numpy()
    #         gt_depth = cv2.imread(one_input[0]["mesh_refined_path"], cv2.IMREAD_ANYDEPTH)
    #         gt_mask = cv2.imread(one_input[0]["mask_path"],cv2.IMREAD_GRAYSCALE)
    #         mirror_mask = gt_mask > 0
    #         if not self.cfg.DEPTH_EST:
    #             instances = one_output[0][0]["instances"]
    #             _, pred_must_correct = draw_gt_bbox(one_input[0]["annotations"] ,gt_mask,instances.pred_anchor_classes)
    #         np_pred_depth = pred_depth.copy()
            
    #         # -------------- refine depth with predict anchor normal ------------
    #         if self.cfg.ANCHOR_CLS:
    #             anchor_normals = np.load(self.cfg.ANCHOR_NORMAL_NYP)
    #             for instance_idx, pred_anchor_normal_class in enumerate(instances.pred_anchor_classes):
    #                 instance_mask = instances.pred_masks[instance_idx].detach().cpu().numpy()
    #                 z = (instance_mask*pred_depth).sum() / (instance_mask>0).sum()
    #                 y = np.where(instance_mask>0)[0].mean() # h
    #                 x = np.where(instance_mask>0)[1].mean() # w
    #                 if pred_anchor_normal_class >= 3:
    #                     continue
    #                 else:
    #                     if self.cfg.ANCHOR_REG:
    #                         plane_normal = anchor_normals[pred_anchor_normal_class] + instances.pred_residuals[instance_idx].detach().cpu().numpy()
    #                     else:
    #                         plane_normal = anchor_normals[pred_anchor_normal_class]
    #                 a, b, c = unit_vector(plane_normal)
                    
    #                 pred_depth = refine_depth_fun(instance_mask, [a, b, c], pred_depth, self.cfg.FOCAL_LENGTH)        


            
    #             current_test_root = ""
    #             raw_input_img_path = one_input[0]["file_name"] 
    #             for one_test_img_root in self.cfg.TEST_IMG_ROOT:
    #                 if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
    #                     current_test_root = one_test_img_root
    #                     break
                
    #         np_pred_depth = np_pred_depth.astype(np.uint16)
    #         depth_p = pred_depth.astype(np.uint16)

    #         if self.cfg.EVAL_SAVE_DEPTH:
    #             gt_mask_path = one_input[0]["mask_path"]
    #             gt_depth_path = one_input[0]["mesh_refined_path"]
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

    #             else:
    #                 print("error some path not exist")
    #             sys.stdout.flush()

    def save_masked_image(self, output_list):
        masked_img_save_folder = os.path.join(self.cfg.EVAL_SAVE_MASKED_IMG, "masked_img")
        os.makedirs(masked_img_save_folder, exist_ok=True)
        output_json_save_path = os.path.join(masked_img_save_folder, "output_info.json")

        estimate_fail = 0
        for one_output, one_input in output_list:
            
            instances = one_output[0][0]["instances"]
            img_path = one_input[0]["file_name"]
            if instances.pred_boxes.tensor.shape[0] <= 0:
                print("######## no detection :", img_path)
                continue

            # img = cv2.imread(img_path) 
            img = cv2.resize(cv2.imread(img_path) , (self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT), 0, 0, cv2.INTER_NEAREST)
            v = Visualizer(img[:, :, ::-1], # chris : init result visulizer
                metadata=MetadataCatalog.get("s3d_mirror_val"), 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                )
            v, colors = v.draw_instance_predictions(instances.to("cpu")) # chris : use result visualizer to show the result
            
            output_img = v.get_image()[:, :, ::-1]
            output_img, predict_correct = draw_gt_bbox(one_input[0]["annotations"] ,output_img,instances.pred_anchor_classes)

            raw_input_img_path = one_input[0]["file_name"] 
            for one_test_img_root in self.cfg.TEST_IMG_ROOT:
                    if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
                        current_test_root = one_test_img_root
                        break
            

            if predict_correct:
                correct_save_folder = os.path.join(masked_img_save_folder, "correct_sample")
                os.makedirs(correct_save_folder, exist_ok=True)

                if "scannet" not in img_save_path:
                    img_save_path = os.path.join(false_save_folder, "masked_img_{}".format(img_path.split("/")[-1]))
                    noraml_vis_save_path = os.path.join(false_save_folder, "normal_vis_{}".format(img_path.split("/")[-1]))
                else:
                    img_save_path = os.path.join(false_save_folder, "masked_img_{}_{}".format(img_path.split("/")[-2], img_path.split("/")[-1]))
                    noraml_vis_save_path = os.path.join(false_save_folder, "normal_vis_{}_{}".format(img_path.split("/")[-2], img_path.split("/")[-1]))
                

                if self.cfg.EVAL_SAVE_NORMAL_VIS:
                    normal_vis_image = get_normal_vis(self.cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)
            else:
                false_save_folder = os.path.join(masked_img_save_folder, "false_sample")
                os.makedirs(false_save_folder, exist_ok=True)

                if "scannet" not in img_save_path:
                    img_save_path = os.path.join(false_save_folder, "masked_img_{}".format(img_path.split("/")[-1]))
                    noraml_vis_save_path = os.path.join(false_save_folder, "normal_vis_{}".format(img_path.split("/")[-1]))
                else:
                    img_save_path = os.path.join(false_save_folder, "masked_img_{}_{}".format(img_path.split("/")[-2], img_path.split("/")[-1]))
                    noraml_vis_save_path = os.path.join(false_save_folder, "normal_vis_{}_{}".format(img_path.split("/")[-2], img_path.split("/")[-1]))
                
                cv2.imwrite(img_save_path, output_img)
                print("masked image saved to :", img_save_path )

                estimate_fail += 1
                if self.cfg.EVAL_SAVE_NORMAL_VIS:
                    normal_vis_image = get_normal_vis(self.cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)


        print("sample may fail : " ,estimate_fail , "sample must corret : ", len(output_list) - estimate_fail)
        print("##################### main output folder #################### {}".format(masked_img_save_folder))
      
    # def visualize_instance_chris(self, time_tag, output_list):
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
    #         for one_test_img_root in self.cfg.TEST_IMG_ROOT:
    #                 if  os.path.abspath(raw_input_img_path.replace(os.path.relpath(raw_input_img_path, one_test_img_root),"")) == os.path.abspath(one_test_img_root):
    #                     current_test_root = one_test_img_root
    #                     break
            

    #         if predict_correct:
    #             correct_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "success_masked_img") )[0]
    #             os.makedirs(correct_save_folder, exist_ok=True)
    #             img_save_path = os.path.join(correct_save_folder, img_path.split("/")[-1])
    #             cv2.imwrite(img_save_path, output_img)
    #             print(  "mask debug image saved to :", img_save_path )
    #             if self.cfg.ANCHOR_CLS and self.cfg.ANCHOR_REG:
    #                 noraml_vis_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "success_noraml_vis") )[0]
    #                 os.makedirs(noraml_vis_save_folder, exist_ok=True)
    #                 noraml_vis_save_path = os.path.join(noraml_vis_save_folder, img_path.split("/")[-1])
    #                 normal_vis_image = get_normal_vis(self.cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)
    #         else:
    #             false_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "fail_masked_img") )[0]

    #             os.makedirs(false_save_folder, exist_ok=True)
    #             img_save_path = os.path.join(false_save_folder, img_path.split("/")[-1])
    #             cv2.imwrite(img_save_path, output_img)
    #             print(  "mask debug image saved to :", img_save_path )
    #             estimate_fail += 1
    #             if self.cfg.ANCHOR_CLS and self.cfg.ANCHOR_REG:
    #                 noraml_vis_save_folder = os.path.split(os.path.join(output_folder, os.path.relpath(img_path, current_test_root)).replace("raw", "fail_noraml_vis") )[0]
    #                 os.makedirs(noraml_vis_save_folder, exist_ok=True)
    #                 noraml_vis_save_path = os.path.join(noraml_vis_save_folder, img_path.split("/")[-1])
    #                 normal_vis_image = get_normal_vis(self.cfg, colors, one_input[0]["annotations"], instances, noraml_vis_save_path)


    #     print("sample may fail : " ,estimate_fail , "sample must corret : ", len(output_list) - estimate_fail)
    #     print("##################### main output folder #################### {}".format(output_folder))

    def eval_seg(self, output_list):

        eval_seg_fun = Mirror_seg_eval(self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            mask_path = one_input[0]["img_path"].replace("raw","instance_mask")
            if not os.path.exists(mask_path):
                continue
            GT_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            # GT_mask = cv2.resize(GT_mask.astype(np.uint8), (self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT), 0, 0, cv2.INTER_NEAREST)
            GT_mask = GT_mask > 0
            pred_mask = np.zeros_like(GT_mask)
            pred_mask = pred_mask.astype(bool)
            if instances.to("cpu").has("pred_masks"):
                for one_pred_mask in instances.to("cpu").pred_masks:
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    pred_mask = pred_mask.numpy().astype(bool)
            eval_seg_fun.compute_and_update_seg_metrics(pred_mask, GT_mask)
        
        
        # print("#################################### {:20} ####################################".format("mean IOU"))
        # print("mean IOU {:.4} for {:6} samples".format(IOU_list.mean(), len(IOU_list)))
        eval_seg_fun.print_seg_score()
        IOU_list, f_measure_list, MAE_list = eval_seg_fun.get_results()
        
        if not self.cfg.EVAL:
            storage = get_event_storage()
            storage.put_scalar("mean IOU",np.mean(IOU_list))
            storage.put_scalar("mean f measure",np.mean(f_measure_list))
            storage.put_scalar("mean MAE",np.mean(MAE_list))

    # def raw_D_refined(self, time_tag, output_list):

    #     from contextlib import redirect_stdout

    #     def unit_vector(vector):
    #         """ Returns the unit vector of the vector.  """
    #         return vector / np.linalg.norm(vector)

    #     def refine_depth_by_mirror_area(instance_mask, plane_normal, np_depth, f=538):
    #         # plane : ax + by + cd + d = 0
    #         h, w = np_depth.shape
    #         a, b, c = plane_normal
    #         offset = (np_depth * instance_mask).sum()/ instance_mask.sum()
    #         py = np.where(instance_mask)[0].mean()
    #         px = np.where(instance_mask)[1].mean()
    #         x0 = (px - w/2) * (offset/ f)
    #         y0 = (py- h/2) * (offset/ f)
    #         d = -(a*x0 + b*y0 + c*offset)
    #         for y in range(h):
    #             for x in range(w):
    #                 if  instance_mask[y][x]:
    #                     n = np.array([a, b, c])
    #                     # plane function : ax + by + cz + d = 0 ---> x = 0 , y = 0 , c = -d/c
    #                     V0 = np.array([0, 0, -d/c])
    #                     P0 = np.array([0,0,0])
    #                     P1 = np.array([(x - w/2), (y - h/2), f ])

    #                     j = P0 - V0
    #                     u = P1-P0
    #                     N = -np.dot(n,j)
    #                     D = np.dot(n,u)
    #                     sI = N / D
    #                     I = P0+ sI*u

    #                     np_depth[y,x] = I[2]
    #         return np_depth

    #     # def refine_depth_by_mirror_border(instance_mask, plane_normal, np_depth, f=538, border_width=50):
    #     #     # plane : ax + by + cd + d = 0
    #     #     h, w = np_depth.shape
    #     #     a, b, c = plane_normal
    #     #     new_mask = cv2.dilate(np.array(instance_mask).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width,border_width)))
    #     #     mirror_border_mask = new_mask - instance_mask
    #     #     offset = (np_depth * mirror_border_mask).sum()/ mirror_border_mask.sum()
    #     #     py = np.where(instance_mask)[0].mean()
    #     #     px = np.where(instance_mask)[1].mean()
    #     #     x0 = (px - w/2) * (offset/ f)
    #     #     y0 = (py- h/2) * (offset/ f)
    #     #     d = -(a*x0 + b*y0 + c*offset)
    #     #     for y in range(h):
    #     #         for x in range(w):
    #     #             if  instance_mask[y][x]:
    #     #                 n = np.array([a, b, c])
    #     #                 # plane function : ax + by + cz + d = 0 ---> x = 0 , y = 0 , c = -d/c
    #     #                 V0 = np.array([0, 0, -d/c])
    #     #                 P0 = np.array([0,0,0])
    #     #                 P1 = np.array([(x - w/2), (y - h/2), f ])

    #     #                 j = P0 - V0
    #     #                 u = P1-P0
    #     #                 N = -np.dot(n,j)
    #     #                 D = np.dot(n,u)
    #     #                 sI = N / D
    #     #                 I = P0+ sI*u

    #     #                 np_depth[y,x] = I[2]
    #     #     return np_depth

    #     def save_txt(save_path, data):
    #         with open(save_path, "w") as file:
    #             for info in data:
    #                 file.write(str(info))
    #                 file.write("\n")
    #         print("txt saved to : ", save_path, len(data))

    #     def read_txt(txt_path):
    #         with open(txt_path, "r") as file:
    #             lines = file.readlines()
    #         return [line.strip() for line in lines]


    #     # ------------- save reinfed depth --------------
    #     refine_depth_output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag ,"refine_depth_fun")
    #     os.makedirs(refine_depth_output_folder, exist_ok=True)
    #     # ------------- save masked image --------------
    #     masked_image_output_folder = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag ,"masked_image")
    #     os.makedirs(masked_image_output_folder, exist_ok=True)
    #     gt_p_np_mask_maskedRaw_list = []
    #     gt_p_np_mask_maskedRaw_txt = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag, "gt_p_np_mask_maskedRaw.txt")
    #     anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
    #     # ------------- save config --------------
    #     yml_save_path = os.path.join("/project/3dlg-hcvc/jiaqit/exp_result/waste/" , time_tag, "test_config.yml")
    #     with open(yml_save_path, 'w') as f:
    #         with redirect_stdout(f): print(self.cfg.dump())
    #     print("############################## Depth Refine Config ##############################")
    #     print("self.cfg.REF_DEPTH_TO_REFINE {} self.cfg.REF_MODE {}".format(self.cfg.REF_DEPTH_TO_REFINE, self.cfg.REF_MODE))
    #     print("self.cfg.VAL_COCO_JSON {}".format(self.cfg.VAL_COCO_JSON))

    #     if os.path.exists(self.cfg.REF_DEPTH_TO_REFINE): # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         depth_list = [item.split()[1] for item in read_txt(self.cfg.REF_DEPTH_TO_REFINE)]

    #         id_depth_path = dict()
    #         for item in depth_list:
    #             id = item.split("/")[-1].split(".")[0]
    #             id_depth_path[id] = item



    #     for i, item in enumerate(output_list):
    #         one_output, one_input = item
    #         instances = one_output[0][0]["instances"]
    #         raw_image_path = one_input[0]["img_path"]
    #         raw_depth_path = one_input[0]["hole_raw_path"] # hole_raw_path
    #         gt_depth_path = one_input[0]["mesh_refined_path"]
    #         mask_path = raw_image_path.replace("raw","instance_mask").replace(".jpg",".png")
    #         if not os.path.exists(mask_path):
    #             mask_path = None
    #         else:
    #             GT_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
    #             GT_mask = [GT_mask > 0] 
            
    #         pred_mask = np.zeros(instances.image_size)
    #         pred_mask = pred_mask.astype(bool)

    #         #################### refine depth using DE / raw-D ####################
    #         # ---------- using DE depth ----------
    #         if self.cfg.REF_MODE == "DE_mirror" or self.cfg.REF_MODE == "DE_border": 
    #             current_depth_id = gt_depth_path.split("/")[-1].split(".")[0]
    #             DE_depth_path = id_depth_path[current_depth_id]
    #             refine_depth_fun = cv2.resize(cv2.imread(DE_depth_path, cv2.IMREAD_ANYDEPTH), (instances.image_size[1], instances.image_size[0]), interpolation=cv2.INTER_NEAREST)
    #         # ---------- using raw-D ----------
    #         elif self.cfg.REF_MODE == "rawD_mirror" or self.cfg.REF_MODE == "rawD_border": 
    #             refine_depth_fun = cv2.resize(cv2.imread(raw_depth_path, cv2.IMREAD_ANYDEPTH), (instances.image_size[1], instances.image_size[0]), interpolation=cv2.INTER_NEAREST)
            
    #         if instances.to("cpu").has("pred_masks"):
    #             for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
                    
    #                 to_refine_area = one_pred_mask.numpy().astype(bool)
    #                 to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
    #                 if to_refine_area.sum() == 0:
    #                     continue
    #                 pred_mask = np.logical_or(pred_mask , one_pred_mask)
    #                 if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
    #                     continue
                    
    #                 pred_normal = anchor_normal[instances.to("cpu").pred_anchor_classes[index]] +  instances.to("cpu").pred_residuals[index].numpy()
    #                 pred_normal = unit_vector(pred_normal)
    #                 # pred_normal = one_input[0]["annotations"][0]["mirror_normal_camera"] # TODO use gt normal to debug
    #                 if "border" in self.cfg.REF_MODE  :
    #                     refine_depth_fun = refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, refine_depth_fun, f=self.cfg.FOCAL_LENGTH ,border_width=self.cfg.REF_BORDER_WIDTH)
    #                 else:
    #                     refine_depth_fun = refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, refine_depth_fun, f=self.cfg.FOCAL_LENGTH)

    #         refine_depth_save_path = os.path.join(refine_depth_output_folder, raw_depth_path.split("/")[-1])
    #         refine_depth_fun[refine_depth_fun<0] = 0


    #         #################### refine depth using DE_offset + raw-D none_mirror ####################
    #         if instances.to("cpu").has("pred_masks") and self.cfg.REF_MODE == "rawD_mirror":
    #             if instances.to("cpu").has("pred_masks"):
    #                 raw_depth = cv2.imread(raw_depth_path, cv2.IMREAD_ANYDEPTH)
    #                 for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
    #                     to_refine_area = one_pred_mask.numpy().astype(bool)
    #                     to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
    #                     pred_mask = np.logical_or(pred_mask , one_pred_mask)
    #                     if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
    #                         continue
    #                     if to_refine_area.sum() == 0:
    #                         continue
    #                     raw_depth[to_refine_area] = refine_depth_fun[to_refine_area]
    #             refine_depth_fun = raw_depth

    #         cv2.imwrite(refine_depth_save_path, np.array(refine_depth_fun, dtype=np.uint16))


    #         #################### save masked image ####################
    #         if self.cfg.EVAL_PLOT_IMAGE:
    #             masked_image_save_path = os.path.join(masked_image_output_folder, raw_image_path.split("/")[-1])
    #             img = cv2.imread(raw_image_path) 
    #             v = Visualizer(img[:, :, ::-1], # chris : init result visulizer
    #                 metadata=MetadataCatalog.get(one_input[0]["dataset_name"]), 
    #                 scale=1, 
    #                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    #                 )
    #             v, colors = v.draw_instance_predictions(instances.to("cpu")) # chris : use result visualizer to show the result
    #             masked_imagae = v.get_image()[:, :, ::-1]
    #             cv2.imwrite(masked_image_save_path, masked_imagae)
    #             gt_p_np_mask_maskedRaw_list.append("{} {} {} {} {}".format(gt_depth_path, refine_depth_save_path, raw_depth_path, mask_path, masked_image_save_path))
    #         else:
    #             gt_p_np_mask_maskedRaw_list.append("{} {} {} {} {}".format(gt_depth_path, refine_depth_save_path, raw_depth_path, mask_path, raw_image_path))
            
    #     save_txt(gt_p_np_mask_maskedRaw_txt, gt_p_np_mask_maskedRaw_list)
            

            
            
