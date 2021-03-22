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




def copy_rgbd_mask_by_id(id_list_path, data_main_folder, dst_folder):
    is_m3d = False
    if data_main_folder.find("m3d") > 0:
        is_m3d = True
    id_list = set(read_txt(id_list_path))

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


    





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument(
        '--id_list_path', default="")
    parser.add_argument(
        '--data_main_folder', default="", help="folder that have raw/ instance_mask/ hole_raw_depth .. as sub-folders")
    parser.add_argument(
        '--dst_folder', default="", help="folder to store new raw/ instance_mask/ hole_raw_depth .. as sub-folders")
    args = parser.parse_args()

    if args.stage == "1":
        copy_rgbd_mask_by_id(args.id_list_path, args.data_main_folder, args.dst_folder)



