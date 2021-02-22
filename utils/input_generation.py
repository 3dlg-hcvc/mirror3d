import os
import cv2
import argparse
from PIL import Image
import time
import numpy as np
import json
import pathlib
import torch
import pycococreatortools
from tqdm import tqdm
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
from annotation.plane_annotation_tool.plane_annotation_tool import *


class Input_Generator(Plane_annotation_tool):

    def __init__(self, dataset_main_folder):
        self.dataset_main_folder = dataset_main_folder


    def get_anchor_norma_info(self, mirror_normal):
        anchor_normal = np.load(self.anchor_normal_path)

        cloest_distance = 100 # init to be a large number
        for i in range(len(anchor_normal)):
            distance_anchor = mirror_normal - anchor_normal[i] 
            distance = np.sqrt(distance_anchor[0]**2 + distance_anchor[1]**2 + distance_anchor[2]**2)
            if distance < cloest_distance:
                cloest_distance = distance
                anchor_normal_class = i #! the last class is background
                anchor_normal_residual = distance_anchor
        
        return anchor_normal_class, list(anchor_normal_residual)

    def convert_list_2_coco(self, cocoFormat_save_path, color_mask_info_mRaw_mRf_hRaw_hRf):

        categories_info = dict()
        mirror_label = 1
        categories_info["supercategory"] = "mirror"
        categories_info["id"] = mirror_label
        categories_info["name"] = "mirror"

        
        # coco annotation id should start from 1
        annotation_id = 1 
        annotations = []
        images = []
        
        for item_index, info in enumerate(tqdm(color_mask_info_mRaw_mRf_hRaw_hRf)):
            raw_img_path,mask_path, info_path , mesh_raw_path, mesh_refined_path, hole_raw_path, hole_refined_path = info
            color_img = Image.open(raw_img_path)
            mesh_refined_img = cv2.imread(mesh_refined_path, cv2.IMREAD_ANYDEPTH)
            h, w = mesh_refined_img.shape
            
            
            raw_img_path_abv = os.path.relpath(raw_img_path, self.dataset_main_folder)
            mesh_raw_path_abv = os.path.relpath(mesh_raw_path, self.dataset_main_folder)
            mesh_refined_path_abv = os.path.relpath(mesh_refined_path, self.dataset_main_folder)
            hole_raw_path_abv = os.path.relpath(hole_raw_path, self.dataset_main_folder)
            hole_refined_path_abv = os.path.relpath(hole_refined_path, self.dataset_main_folder)
            # ---------- coco images ------------
            image = {
                    "id": item_index+1, # same as "image_id"
                    "height": h,
                    "width" : w,
                    "img_path": raw_img_path_abv,
                    "mesh_raw_path": mesh_raw_path_abv,
                    "mesh_refined_path": mesh_refined_path_abv,
                    "hole_raw_path": hole_raw_path_abv,
                    "hole_refined_path": hole_refined_path_abv,
                }
            if image not in images:
                images.append(image)
            # ---------- coco annotation ------------
            if info_path == None:
                continue
                annotation["id"] = annotation_id
                annotation["image_id"] = item_index+1,
                annotation["category_id"] = 1
                annotation["iscrowd"] = 0
                annotation["area"] = 1
                annotation["bbox"] = []
                annotation["segmentation"] = [[]]
                annotation["width"] = w
                annotation["height"] = h

                annotation["mirror_normal_camera"] = unit_vector(img_info[str(instance_index)]["mirror_normal"]).tolist()
                anchor_normal_class, anchor_normal_residual = get_anchor_norma_info(args, annotation["mirror_normal_camera"], self.anchor_normal_path)
                annotation["anchor_normal_class"] = anchor_normal_class
                annotation["anchor_normal_residual"] = anchor_normal_residual
                annotation["depth_path"] = mesh_refined_path_abv
                annotation["mesh_refined_path"] = mesh_refined_path_abv
                annotation["hole_refined_path"] = hole_refined_path_abv
                annotation["mesh_raw_path"] = mesh_raw_path_abv
                annotation["hole_raw_path"] = hole_raw_path_abv
                annotation["image_path"] = raw_img_path_abv
                annotation["instance_index"] = str(instance_index)
                annotation["mask_path"] = mask_path_abv
                annotations.append(annotation)
                annotation_id += 1
            else:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
                img_info = read_json(info_path)
                mask_path_abv = os.path.relpath(mask_path, self.dataset_main_folder)
                for instance_index in np.unique(mask_img):
                    if instance_index == 0: # background
                        continue
                    if str(instance_index) not in img_info:
                        print("no annotation for instance {} in {}".format(instance_index, color_img), mask_path)
                    else:
                        ground_truth_binary_mask = mask_img.copy()
                        ground_truth_binary_mask[ground_truth_binary_mask!=instance_index] = 0
                        category_info = {'id': mirror_label, 'is_crowd': 0}
                        annotation = pycococreatortools.create_annotation_info(
                                        annotation_id, item_index+1, category_info, ground_truth_binary_mask,
                                        color_img.size, tolerance=2)
                        annotation["mirror_normal_camera"] = unit_vector(img_info[str(instance_index)]["mirror_normal"]).tolist()
                        anchor_normal_class, anchor_normal_residual = self.get_anchor_norma_info(annotation["mirror_normal_camera"])
                        annotation["anchor_normal_class"] = anchor_normal_class
                        annotation["anchor_normal_residual"] = anchor_normal_residual
                        annotation["depth_path"] = mesh_refined_path_abv
                        annotation["mesh_refined_path"] = mesh_refined_path_abv
                        annotation["hole_refined_path"] = hole_refined_path_abv
                        annotation["mesh_raw_path"] = mesh_raw_path_abv
                        annotation["hole_raw_path"] = hole_raw_path_abv
                        annotation["image_path"] = raw_img_path_abv
                        annotation["instance_index"] = str(instance_index)
                        annotation["mask_path"] = mask_path_abv
                        annotations.append(annotation)
                        annotation_id += 1

        coco_format_output = dict()
        coco_format_output["annotations"] = annotations
        coco_format_output["images"] = images
        coco_format_output["info"] = info = {
                                "description": self.dataset_main_folder,
                                "date_created": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                            }
        coco_format_output["categories"] = [categories_info]

        save_json(cocoFormat_save_path, coco_format_output)  # TODO add back after debug

    def color_img_contain_mirror(self, mirror_tag_list, file_path):
        file_path = file_path.split(".")[0]
        file_path_list = file_path.split("/")
        for tags in mirror_tag_list:
            if all([(i in file_path_list) for i in tags]):
                return True
        return False

    def path_in_set(self, set_tags, file_path):
        file_path = file_path.split(".")[0]
        file_path_list = file_path.split("/")
        for item in set_tags:
            if item in file_path_list:
                return True
        return False
                
    def generate_coco_main(self, train_tag_txt, val_tag_txt, test_tag_txt, source_color_txt, color_mirror_folder, dataset_main_folder, kmeans_normal_path, output_main_folder):
        """
        # ! color_mirror_folder is the color image folder path ***/raw
        #! require scannet mirror data store in one folder : and named like "scene0612_01_000835.png"
        # ! under m3d dataset main folder should be : matterport_render_depth  undistorted_color_images  undistorted_depth_images
        """

        self.anchor_normal_path = kmeans_normal_path
        self.dataset_main_folder = dataset_main_folder
        
        dataset_name = ""
        if color_mirror_folder.find("nyu") > 0:
            dataset_name = "nyu"
        elif color_mirror_folder.find("m3d") > 0:
            dataset_name = "m3d"
        elif color_mirror_folder.find("scannet") > 0:
            dataset_name = "scannet"

        if os.path.exists(train_tag_txt):
            train_tags = read_txt(train_tag_txt)
        if  os.path.exists(test_tag_txt):
            test_tags = read_txt(test_tag_txt)
        if os.path.exists(val_tag_txt):
            val_tags = read_txt(val_tag_txt)


        
        if not os.path.exists(color_mirror_folder):
            print(" {} path not exist !!".format(color_mirror_folder))
            return

        # --- get mirror tag list to fiter out positive sample -----
        mirror_tag_list = [] 
        for mirror_color_img_name in os.listdir(color_mirror_folder):
            if dataset_name == "scannet": 
                mirror_tag_list.append([mirror_color_img_name.rsplit("_",1)[0], mirror_color_img_name.rsplit("_",1)[1].split(".")[0]])
            else:
                mirror_tag_list.append([mirror_color_img_name.split(".")[0]])


        train_color_mask_info_mRaw_mRf_hRaw_hRf = []
        test_color_mask_info_mRaw_mRf_hRaw_hRf = []
        val_color_mask_info_mRaw_mRf_hRaw_hRf = []

        # ------ get non-mirror information --------
        if os.path.exists(source_color_txt):
            non_color_depth_list = read_txt(source_color_txt)
            for one_color_path in non_color_depth_list:
                if not self.color_img_contain_mirror(mirror_tag_list, one_color_path):
                    if dataset_name == "m3d":
                        one_hole_raw_path = one_color_path.replace("undistorted_color_images", "undistorted_depth_images").replace(".jpg", ".png")
                        one_hole_raw_path = rreplace(one_hole_raw_path, "i", "d")
                        one_mesh_raw_path = one_hole_raw_path.replace("undistorted_depth_images", "matterport_render_depth")
                        one_mesh_raw_path = rreplace(one_mesh_raw_path, "matterport_render_depth", "mesh_images")
                        if os.path.exists(one_hole_raw_path) and os.path.exists(one_mesh_raw_path) and os.path.exists(one_color_path):
                            one_mesh_refined_path = one_mesh_raw_path
                            one_hole_refined_path = one_hole_raw_path
                            one_path_list = [one_color_path, None, None, one_mesh_raw_path, one_mesh_refined_path, one_hole_raw_path, one_hole_refined_path]
                        else:
                            continue
                    else:
                        one_hole_raw_path = one_color_path.replace("color", "depth").replace(".jpg", ".png")
                        one_mesh_raw_path = one_mesh_refined_path = one_hole_refined_path = one_hole_raw_path
                        if os.path.exists(one_hole_raw_path) and os.path.exists(one_mesh_raw_path):
                            one_path_list = [one_color_path, None, None, one_mesh_raw_path, one_mesh_refined_path, one_hole_raw_path, one_hole_refined_path]
                        else:
                            continue
                    
                    if self.path_in_set(train_tags, one_color_path) and train_tag_txt:
                        train_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
                    elif self.path_in_set(test_tags, one_color_path) and test_tag_txt:
                        test_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
                    elif os.path.exists(val_tag_txt):
                        val_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
                else:
                    pass

        # ------ get mirror information --------
        for mirror_color_img_name in os.listdir(color_mirror_folder):
            one_color_path = os.path.join(color_mirror_folder, mirror_color_img_name)
            one_mask_path = one_color_path.replace("raw", "instance_mask") # ! require all image suffix is .png
            one_info_path = one_color_path.replace("raw", "img_info").split(".")[0] + ".json"

            if dataset_name == "m3d":
                one_mesh_raw_path = rreplace(one_color_path.replace("raw", "mesh_raw_depth"), "i", "d")
                one_mesh_refined_path = rreplace(one_color_path.replace("raw", "mesh_refined_depth"), "i", "d")
                one_hole_raw_path = rreplace(one_color_path.replace("raw", "hole_raw_depth"), "i", "d")
                one_hole_refined_path = rreplace(one_color_path.replace("raw", "hole_refined_depth"), "i", "d")
            else:
                one_hole_raw_path = one_color_path.replace("raw", "hole_raw_depth")
                one_hole_refined_path = one_color_path.replace("raw", "hole_refined_depth")
                one_mesh_raw_path = one_hole_raw_path
                one_mesh_refined_path = one_hole_refined_path

            one_path_list = [one_color_path, one_mask_path, one_info_path, one_mesh_raw_path, one_mesh_refined_path, one_hole_raw_path, one_hole_refined_path]


         
            if all([os.path.exists(i) for i in one_path_list]):
                if self.path_in_set(train_tags, one_color_path) and train_tag_txt:
                    train_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
                elif self.path_in_set(test_tags, one_color_path) and test_tag_txt:
                    test_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
                elif os.path.exists(val_tag_txt):
                    val_color_mask_info_mRaw_mRf_hRaw_hRf.append(one_path_list)
            else:
                # some path don't exist
                continue 

        if not os.path.exists(source_color_txt):
            json_tag = "{}_only_mirror".format(kmeans_normal_path.split("/")[-1].split(".")[0])
        else:
            json_tag = "{}_with_non_mirror".format(kmeans_normal_path.split("/")[-1].split(".")[0])
        json_output_folder = os.path.join(output_main_folder, dataset_name, json_tag)
        os.makedirs(json_output_folder, exist_ok=True)
        normal_num = np.load(kmeans_normal_path).shape[0]

        cocoFormat_save_path = os.path.join(json_output_folder , "pos_test_normalFormat_{}_normal.json".format(normal_num))
        self.convert_list_2_coco(cocoFormat_save_path, test_color_mask_info_mRaw_mRf_hRaw_hRf)
        if os.path.exists(val_tag_txt):
            cocoFormat_save_path = os.path.join(json_output_folder , "pos_val_normalFormat_{}_normal.json".format(normal_num))
            self.convert_list_2_coco(cocoFormat_save_path, val_color_mask_info_mRaw_mRf_hRaw_hRf)
        cocoFormat_save_path = os.path.join(json_output_folder , "pos_train_normalFormat_{}_normal.json".format(normal_num))
        self.convert_list_2_coco(cocoFormat_save_path, train_color_mask_info_mRaw_mRf_hRaw_hRf)
            
    

    def only_detection_2_coco(self, color_img_list_txt, coco_save_path, dataset_main_folder):
        self.dataset_main_folder = dataset_main_folder
        coco_save_folder = os.path.split(coco_save_path)[0]
        os.makedirs(coco_save_folder, exist_ok=True)
        color_img_list =  read_txt(color_img_list_txt)

        categories_info = dict()
        mirror_label = 1
        categories_info["supercategory"] = "mirror"
        categories_info["id"] = mirror_label
        categories_info["name"] = "mirror"

        
        # coco annotation id should start from 1
        annotation_id = 1 
        annotations = []
        images = []
        unique_id = 1
        for item_index, raw_img_path in enumerate(tqdm(color_img_list)):

            color_img = Image.open(raw_img_path)
            mask_path = raw_img_path.replace("raw", "instance_mask")
            h, w = color_img.size
            raw_img_path_abv = os.path.relpath(raw_img_path, self.dataset_main_folder)
            if not os.path.exists(mask_path) or not os.path.exists(raw_img_path):
                continue

            # ---------- coco images ------------
            image = {
                    "id": unique_id, # same as "image_id"
                    "height": h,
                    "width" : w,
                    "img_path": raw_img_path_abv,
                }
            
            # ---------- coco annotation ------------


            mask_img = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            mask_path_abv = os.path.relpath(mask_path, self.dataset_main_folder)
            for instance_index in np.unique(mask_img):
                if instance_index == 0: # background
                    continue
                ground_truth_binary_mask = mask_img.copy()
                ground_truth_binary_mask[ground_truth_binary_mask!=instance_index] = 0
                category_info = {'id': mirror_label, 'is_crowd': 0}
                annotation = pycococreatortools.create_annotation_info(
                                annotation_id, unique_id, category_info, ground_truth_binary_mask,
                                color_img.size, tolerance=2)
                try:
                    annotation["image_path"] = raw_img_path_abv
                except:
                    # all image black
                    continue
                annotation["instance_index"] = str(instance_index)
                annotation["mask_path"] = mask_path_abv
                annotations.append(annotation)
                annotation_id += 1

            if image not in images:
                images.append(image)

            unique_id += 1

        coco_format_output = dict()
        coco_format_output["annotations"] = annotations
        coco_format_output["images"] = images
        coco_format_output["info"] = info = {
                                "description": self.dataset_main_folder,
                                "date_created": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                            }
        coco_format_output["categories"] = [categories_info]

        save_json(coco_save_path, coco_format_output)  



if __name__ == "__main__":
    # TODO debug this script on NYUV2's & matterport's data



    parser = argparse.ArgumentParser(description='Get Setting :D')
    # -------- parameter for generating detection + depth + normal information --------
    parser.add_argument( 
        '--train_tag_txt', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/train_id.txt", type=str, help="training image's unique tag") 
    parser.add_argument( 
        '--test_tag_txt', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/test_id.txt", type=str, help="testing image's unique tag") 
    parser.add_argument( 
        '--val_tag_txt', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/val_id.txt", type=str, help="validation image's unique tag") 
    parser.add_argument( # ! this require to put the image under folder [dataset_name]/[with_mirror/no_mirror]/[precise/coarse]/raw
        '--color_mirror_folder', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/m3d/with_mirror/coarse/raw", type=str, help="color image folder path") 
    parser.add_argument( 
        '--kmeans_normal_path', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/mirror_normal/m3d_kmeans_normal_10.npy", type=str, help="keams normal .npy file path") 
    parser.add_argument( 
        '--output_main_folder', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/coco_input", type=str, help="coco_format.json output main folder") 
    # -------- parameter for generating only detection information --------
    parser.add_argument( 
        '--output_coco_file_path', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/coco_input/MSD_PMD/msd_pmd_train.json", type=str, help="all color images's absolute path from original dataset") 
    # -------- commonly use parameter --------
    parser.add_argument( # ! the absolute path of  Mirror3D_dataset/[dataset_name]
        '--dataset_main_folder', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset", type=str, help="dataset main folder") 
    parser.add_argument( 
        '--source_color_txt', default="/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/msd_pmd_train_raw.txt", type=str, help="all color images's absolute path from original dataset") 
    args = parser.parse_args()
    
    fun = Input_Generator()
    # fun.generate_coco_main(args.train_tag_txt, args.val_tag_txt, args.test_tag_txt, args.source_color_txt, args.color_mirror_folder, args.dataset_main_folder, args.kmeans_normal_path, args.output_main_folder)

    fun.only_detection_2_coco(args.source_color_txt, args.output_coco_file_path, args.dataset_main_folder)





    # train_tag_txt = "/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/summary/train_test_split/train.txt"
    # test_tag_txt = "/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/summary/train_test_split/test.txt"
    # val_tag_txt = None
    # source_color_txt = None
    # # source_color_txt = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/nyu_source_color_list.txt"
    # color_mirror_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/nyu/with_mirror/refined/raw"
    # kmeans_normal_path = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/mirror_normal/m3d_kmeans_normal_10.npy"
    # output_main_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/coco_input"
    # dataset_main_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/nyu"
    # train_tag_txt = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/train_id.txt"
    # test_tag_txt = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/test_id.txt"
    # val_tag_txt = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/other_info/m3d/val_id.txt"
    # source_color_txt = None
    # color_mirror_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/m3d/with_mirror/coarse/raw"
    # kmeans_normal_path = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/mirror_normal/m3d_kmeans_normal_10.npy"
    # output_main_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/coco_input"
    # dataset_main_folder = "/project/3dlg-hcvc/jiaqit/Mirror3D_dataset/m3d"