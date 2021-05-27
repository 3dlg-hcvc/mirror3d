import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
import sys
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
import json
import shutil
from annotation.plane_annotation.plane_annotation_tool import *
from tqdm import tqdm
 

def reformat_json(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for one_json_name in os.listdir(input_folder):
        src_json_path =  os.path.join(input_folder, one_json_name)
        ori_info = read_json(src_json_path)
        new_info = []
        if len(ori_info) > 1:
            for item in ori_info.items():
                new_one_info = dict()
                new_one_info["plane"] = item[1]["plane_parameter"]
                new_one_info["normal"] = list(unit_vector(item[1]["mirror_normal"]))
                R, G, B = [int(i) for i in item[0].split("_")]
                new_one_info["mask_id"] = '%02x%02x%02x' % (R, G, B)
                new_info.append(new_one_info)
            json_save_path = os.path.join(output_folder, one_json_name)
            save_json(json_save_path,new_info)



        

def reformat_json2(input_folder, output_folder):
    test = read_json("waste/test.json")
    for one_json_name in os.listdir(input_folder):
        src_json_path =  os.path.join(input_folder, one_json_name)
        ori_info = read_json(src_json_path)
        new_info = dict()
        if len(ori_info) == 1:
            continue
        for item in ori_info.items():
            new_one_info = dict()
            new_one_info["plane"] = item[1]["plane_parameter"]
            new_one_info["normal"] = list(unit_vector(item[1]["mirror_normal"]))
            R, G, B = [int(i) for i in item[0].split("_")]
            new_one_info["mask_id"] = '%02x%02x%02x' % (R, G, B)
            new_info['%02x%02x%02x' % (R, G, B)] = new_one_info

        save_json("waste/test.json",new_info)
        break



def get_delta_image(refinedD_input_folder, rawD_input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    for one_img_name in tqdm(os.listdir(refinedD_input_folder)):
        delta_img_save_path = os.path.join(output_folder, one_img_name)
        if os.path.exists(delta_img_save_path):
            continue
        refD_img_path = os.path.join(refinedD_input_folder, one_img_name)
        rawD_img_path = os.path.join(rawD_input_folder, one_img_name)
        delta_img = cv2.imread(refD_img_path, cv2.IMREAD_ANYDEPTH) - cv2.imread(rawD_img_path, cv2.IMREAD_ANYDEPTH)
        cv2.imwrite(delta_img_save_path, delta_img.astype(np.uint16))
    print("delta image saved to : ", output_folder)


def update_coco_json(ori_json_folder):
    for one_name in os.listdir(ori_json_folder):
        ori_json_file_path = os.path.join(ori_json_folder, one_name)
        ori_info = read_json(ori_json_file_path)
        new_info = dict()
        new_info["annotations"] = []
        new_info["images"] = []
        new_info["categories"] = ori_info["categories"]
        for item in ori_info["annotations"]:
            new_item = item.copy()
            new_item.pop("depth_path")
            new_item.pop("mesh_refined_path")
            new_item.pop("hole_refined_path")
            new_item.pop("mesh_raw_path")
            new_item.pop("hole_raw_path")
            R, G, B = [int(i) for i in ori_info["instance_tag"].split("_")]
            new_item["mask_id"] = '%02x%02x%02x' % (R, G, B)
            new_item["refined_sensorD"] = item["hole_refined_path"].replace("with_mirror/precise/raw","mirror_color_image").replace(".png",".jpg")
            new_item["raw_sensorD"] = item["hole_raw_path"].replace("with_mirror/precise/hole_raw_depth","raw_sensorD_precise")
            new_item["refined_meshD"] = item["mesh_refined_path"]
            new_item["raw_meshD"] = item["mesh_raw_path"]
            new_info["annotations"].append(new_item)
        for item in ori_info["images"]:
            new_item = item.copy()
            new_item.pop("mesh_refined_path")
            new_item.pop("hole_refined_path")
            new_item.pop("mesh_raw_path")
            new_item.pop("hole_raw_path")
            new_item["refined_sensorD"] = item["hole_refined_path"]
            new_item["raw_sensorD"] = item["hole_raw_path"]
            new_item["refined_meshD"] = item["mesh_refined_path"]
            new_item["raw_meshD"] = item["mesh_raw_path"]
        # ori_lines = read_txt(json_file_path)
        # ori_string = "mirror_instance_mask_precise"
        # to_replace_string = "mirror_instance_mask_precise/"
        # new_lines = [line.replace(ori_string, to_replace_string) for line in ori_lines]
        # save_txt(json_file_path, new_lines)
        # json_temp = read_json(json_file_path)
        # save_json(json_file_path, json_temp)

def generate_symlinks_txt_mp3d():
    output_lines = []
    all_id = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/mp3d_all.txt")
    all_color_list = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/m3d_color.txt")
    save_path = "/local-scratch/jiaqit/exp/Mirror3D/dataset/metadata/m3d_symlink.txt"
    for one_path in all_color_list:
        id = one_path.split("/")[-1].split(".")[0]
        if id in all_id:
            color_to_link = "mirror_color_images/{}".format(one_path.split("/")[-1])
            sensorD_to_link = "raw_sensorD/{}".format(rreplace(one_path.replace("color","depth").replace("jpg","png"),"i","d").split("/")[-1])
            meshD_to_link = "raw_meshD/{}".format(rreplace(one_path.replace("color","depth").replace("jpg","png"),"i","d").split("/")[-1])
            output_lines.append("{} {}".format(one_path, color_to_link))
            output_lines.append("{} {}".format(rreplace(one_path.replace("color","depth").replace("jpg","png"),"i","d"), sensorD_to_link))
            hole_raw_path = rreplace(one_path.replace("color","depth").replace("jpg","png"),"i","d")
            ori_meshD_path = ((rreplace(hole_raw_path, "undistorted_depth_images", "mesh_images")).replace("undistorted_depth_images", "matterport_render_depth")).split(".")[0] + "_mesh_depth.png"
            output_lines.append("{} {}".format(ori_meshD_path, meshD_to_link))
    save_txt(save_path, output_lines)

def generate_symlinks_txt_nyu():
    output_lines = []
    all_id = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/nyu_all.txt")
    save_path = "/local-scratch/jiaqit/exp/Mirror3D/metadata/nyu_symlink.txt"
    all_color_list = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/nyu_color.txt")
    for one_path in all_color_list:
        id = one_path.split("/")[-1].split(".")[0]
        if id in all_id:
            color_to_link = "mirror_color_images/{}".format(one_path.split("/")[-1])
            sensorD_to_link = "raw_sensorD/{}".format(one_path.replace("color","depth").replace("jpg","png").split("/")[-1])
            output_lines.append("{} {}".format(one_path, color_to_link))
            output_lines.append("{} {}".format(one_path.replace("color","depth").replace("jpg","png"), sensorD_to_link))
    save_txt(save_path, output_lines)

def generate_symlinks_txt_scannet():
    output_lines = []
    all_id = [i.split(".")[0] for i in read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/scannet_all.txt")]
    save_path = "/local-scratch/jiaqit/exp/Mirror3D/metadata/scannet_symlink.txt"
    all_color_list_25k = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/scannet_color_25k.txt")
    all_color_list = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/scannet_color.txt")
    for one_id in all_id:
        path_one = "scannet_frames_25k/{}/color/{}.jpg".format(one_id.rsplit("_", 1)[0], one_id.rsplit("_", 1)[1])
        path_two = "scannet_extracted/raw_image/{}/color/{}.jpg".format(one_id.rsplit("_", 1)[0], str(int(one_id.rsplit("_", 1)[1])))
        if path_two not in all_color_list and path_one not in all_color_list_25k:
            print(one_id)
        if path_two in all_color_list:
            color_to_link = "mirror_color_images/{}.jpg".format(one_id)
            sensorD_to_link = "raw_sensorD/{}.png".format(one_id)
            output_lines.append("{} {}".format(path_two, color_to_link))
            output_lines.append("{} {}".format(path_two.replace("color","depth").replace("jpg","png"), sensorD_to_link))
        elif path_one in all_color_list_25k:
            color_to_link = "mirror_color_images/{}.jpg".format(one_id)
            sensorD_to_link = "raw_sensorD/{}.png".format(one_id)
            output_lines.append("{} {}".format(path_one, color_to_link))
            output_lines.append("{} {}".format(path_one.replace("color","depth").replace("jpg","png"), sensorD_to_link))
    # for one_path in all_color_list:
    #     id = "{}_{}".format(one_path.split("/")[-3], one_path.split("/")[-1].split(".")[0].zfill(6))
    #     if id in all_id:
            # color_to_link = "mirror_color_images/{}.jpg".format(id)
            # sensorD_to_link = "raw_sensorD/{}.png".format(id)
            # output_lines.append("{} {}".format(one_path, color_to_link))
            # output_lines.append("{} {}".format(one_path.rep ace("color","depth").replace("jpg","png"), sensorD_to_link))

    # save_txt(save_path, output_lines)

    save_txt(save_path, output_lines)


def sort_mp3d_data():
    all_color_img_list = read_txt("/project/3dlg-hcvc/mirrors/www/dataset_release/temp/m3d_color.txt")
    unzip_main_folder = "/project/3dlg-hcvc/mirrors/www/dataset_pack/mp3d"
    id_sceneID = dict()

    for one_path in all_color_img_list:
        scene_id = one_path.split("/")[-3]
        img_id = one_path.split("/")[-1].split(".")[0]
        id_sceneID[img_id] = scene_id


    for one_folder_name in os.listdir(unzip_main_folder):
        one_folder_path = os.path.join(unzip_main_folder, one_folder_name)
        for one_file_name in tqdm(os.listdir(one_folder_path)):
            src_path = os.path.join(one_folder_path, one_file_name)
            if "png" not in src_path and "jpg" not in src_path and  "json" not in src_path:
                continue
            img_id = one_file_name.split(".")[0]
            if "delta" in src_path:
                scene_id = id_sceneID[rreplace(img_id, "d", "i")]
            else:
                scene_id = id_sceneID[img_id]
            dst_folder = os.path.join(one_folder_path, scene_id)
            os.makedirs(dst_folder,exist_ok=True)
            shutil.move(src_path, dst_folder)
        print("finish", one_folder_path)
            



def sort_scannet_data():
    unzip_main_folder = "/project/3dlg-hcvc/mirrors/www/dataset_pack/scannet"

    for one_folder_name in os.listdir(unzip_main_folder):
        one_folder_path = os.path.join(unzip_main_folder, one_folder_name)
        for one_file_name in tqdm(os.listdir(one_folder_path)):
            src_path = os.path.join(one_folder_path, one_file_name)
            if "png" not in src_path and "jpg" not in src_path and  "json" not in src_path:
                continue
            img_id = one_file_name.split(".")[0]
            scene_id = img_id.rsplit("_", 1)[0]
            img_frame_name = "{}.{}".format(str(int(img_id.rsplit("_", 1)[1])),one_file_name.split(".")[1]) 
            dst_folder = os.path.join(one_folder_path, scene_id)
            os.makedirs(dst_folder,exist_ok=True)
            dst_path = os.path.join(dst_folder, img_frame_name)
            shutil.move(src_path, dst_path)
        print("finish", one_folder_path)
            



    
if __name__ == "__main__":
    # json_file_path = "/project/3dlg-hcvc/mirrors/www/dataset_release/network_input_json/nyu"
    # update_coco_json(json_file_path)
    # input_folder = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/img_info"
    # output_folder = "waste"
    # reformat_json(input_folder, output_folder)

    # refinedD_input_folder = "/project/3dlg-hcvc/mirrors/www/dataset_release/mp3d/refined_sensorD_coarse"
    # rawD_input_folder = "/project/3dlg-hcvc/mirrors/www/dataset_release/mp3d/raw_sensorD"
    # output_folder = "/project/3dlg-hcvc/mirrors/www/dataset_release/mp3d/delta_image_coarse"
    # get_delta_image(refinedD_input_folder, rawD_input_folder, output_folder)


    ############# generate symlinks 
    # generate_symlinks_txt_mp3d()
    # generate_symlinks_txt_scannet()

    ########## scort mp3d data into scene_image_id 
    sort_mp3d_data()
    # sort_scannet_data()

