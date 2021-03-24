import sys
import os
import time
import json
import random
import time
import numpy as np
from utils.algorithm import *
from utils.general_utlis import *
from PIL import Image, ImageDraw
import shutil
import argparse




def copy_rgbd_mask_by_id(id_list, data_main_folder, dst_folder):
    is_m3d = False
    if data_main_folder.find("m3d") > 0:
        is_m3d = True
    

    raw_folder = os.path.join(data_main_folder, "raw")
    instance_mask_folder = os.path.join(data_main_folder, "instance_mask")
    hole_raw_depth_folder = os.path.join(data_main_folder, "hole_raw_depth")

    if is_m3d:
        mesh_raw_depth_folder = os.path.join(data_main_folder, "mesh_raw_depth")


    for one_id in id_list:
        raw_path = os.path.join(raw_folder, "{}.png".format(one_id))
        instance_mask_path = os.path.join(instance_mask_folder, "{}.png".format(one_id))
        
        if is_m3d:
            mesh_raw_depth_path = os.path.join(mesh_raw_depth_folder, "{}.png".format(rreplace(one_id,"i","d")))
            hole_raw_depth_path = os.path.join(hole_raw_depth_folder, "{}.png".format(rreplace(one_id,"i","d")))
        else:
            hole_raw_depth_path = os.path.join(hole_raw_depth_folder, "{}.png".format(one_id))
            mesh_raw_depth_path = hole_raw_depth_path

        if os.path.exists(raw_path) and os.path.exists(instance_mask_path) and os.path.exists(hole_raw_depth_path) and os.path.exists(mesh_raw_depth_path):
            for index, one_src in enumerate([raw_path, instance_mask_path, hole_raw_depth_path, mesh_raw_depth_path]):
                one_dst_path = one_src.replace(data_main_folder, dst_folder)
                one_dst_folder = os.path.split(one_dst_path)[0]
                os.makedirs(one_dst_folder, exist_ok=True)
                shutil.copy(one_src, one_dst_folder)
                # print("copying {} to {}".format(one_dst_path, one_dst_folder))
                if not is_m3d and index == 2:
                    break

        else:
            print("STH doesn't exist {} {} {} {} {} {} {} {}".format(os.path.exists(raw_path), raw_path, \
                                                            os.path.exists(instance_mask_path), instance_mask_path, \
                                                            os.path.exists(hole_raw_depth_path), hole_raw_depth_path, \
                                                            os.path.exists(mesh_raw_depth_path), mesh_raw_depth_path))


def get_error_sample_rgbd_mask(data_main_folder, dst_folder, current_raw_folder):

    all_raw_folder = os.path.join(data_main_folder, "raw")
    current_raw_name_list = [i for i in os.listdir(current_raw_folder)]
    current_raw_name_list.sort()

    all_raw_name_list = [i for i in os.listdir(all_raw_folder)]
    all_raw_name_list.sort()

    lack_sample_name = list(set(all_raw_name_list) - set(current_raw_name_list))
    lack_sample_name.sort()

    lack_id_list = [i.split(".")[0] for i in lack_sample_name]
    lack_id_list = set(lack_id_list)

    copy_rgbd_mask_by_id(lack_id_list, data_main_folder, dst_folder)



def get_diff(path_one, path_two):
    lines_one = read_txt(path_one)
    print("{} all num : {} unique num : {}".format(path_one, len(lines_one), len(set(lines_one))))

    lines_two = read_txt(path_two)
    print("{} all num : {} unique num : {}".format(path_two, len(lines_two), len(set(lines_two))))


    print(set(lines_one) - set(lines_two))

    print(set(lines_two) - set(lines_one))
    

def count_unique_line_num(txt_path):
    lines = read_txt(txt_path)
    print("all num : {} unique num : {}".format(len(lines), len(set(lines))))


def check_invalid():
    invalid = read_txt('/project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise/anno_progress/error_id.txt')

    raw_folder = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/only_mask/precise/raw"
    print("invalid id num {}".format(len(set(invalid))))
    for one_name in os.listdir(raw_folder):
        one_path = os.path.join(raw_folder, one_name)
        one_id = one_name.split(".")[0]
        if one_id not in invalid:
            print(one_path)

        # def is_complete(color_list, folder_path):
        #     for index, one_color_path in enumerate(color_list):
                


        # self.color_img_list = [os.path.join(self.data_main_folder, "raw", i) for i in os.listdir(os.path.join(self.data_main_folder, "raw"))]
        # self.color_img_list.sort()

        # if self.is_matterport3d:
        #     mesh_refined_depth_folder = os.path.join(self.data_main_folder, "mesh_refined_depth")
        
        # hole_refined_depth_folder = os.path.join(self.data_main_folder, "hole_refined_depth_folder")

def check_complete(raw_folder="", check_folder="", img_info_folder="", is_m3d_depth=False):
    color_img_list =  [os.path.join(raw_folder, i) for i in os.listdir(os.path.join(raw_folder))]
    color_img_list.sort()
    

    existing_id_list = [i.split(".")[0] for i in os.listdir(check_folder)]


    for index, one_color_path in enumerate(color_img_list):
        one_info_path = one_color_path.replace("raw","img_info").replace(".png", ".json")
        one_info = read_json(one_info_path)
        one_color_name = one_color_path.split("/")[-1].split(".")[0]
        # import pdb; pdb.set_trace()

        # if check name = ***_idx_***
        if os.path.exists(img_info_folder):
            for item in one_info.items():
                if is_m3d_depth:
                    should_exist_id = "{}_idx_{}".format(rreplace(one_color_name,"i","d"), item[0])
                else:
                    should_exist_id = "{}_idx_{}".format(one_color_name, item[0])
                if should_exist_id not in existing_id_list:
                    print("index {} : {} not exist".format(index, should_exist_id))
        else:
            if is_m3d_depth:
                should_exist_id = rreplace(one_color_name,"i","d")
            else:
                should_exist_id = one_color_name
            if should_exist_id not in existing_id_list:
                print("index {} : {} not exist".format(index, should_exist_id))


        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument(
        '--txt_path', default="")
    parser.add_argument(
        '--id_list_path', default="")
    parser.add_argument(
        '--data_main_folder', default="", help="folder that have raw/ instance_mask/ hole_raw_depth .. as sub-folders")
    parser.add_argument(
        '--dst_folder', default="", help="folder to store new raw/ instance_mask/ hole_raw_depth .. as sub-folders")
    parser.add_argument(
        '--current_raw_folder', default="", help="raw folder that contains color images")
    parser.add_argument(
        '--raw_folder', default="")
    parser.add_argument(
        '--check_folder', default="")
    parser.add_argument(
        '--img_info_folder', default="")
    parser.add_argument('--is_m3d_depth',action='store_true')
    args = parser.parse_args()

    if args.stage == "1":
        id_list = set(read_txt(args.id_list_path))
        copy_rgbd_mask_by_id(id_list, args.data_main_folder, args.dst_folder)
    elif args.stage == "2":
        get_error_sample_rgbd_mask(args.data_main_folder, args.dst_folder, args.current_raw_folder)
    elif args.stage == "3":
        count_unique_line_num(args.txt_path)
    elif args.stage == "4":
        paths = args.txt_path.split(",")
        print(paths)
        get_diff(paths[0],paths[1])
    elif args.stage == "5":
        check_invalid()
    elif args.stage == "6":
        check_complete(args.raw_folder, args.check_folder, args.img_info_folder, args.is_m3d_depth)


