import argparse
import os
from mirror3d.utils.algorithm import *
from mirror3d.utils.general_utils import *
from mirror3d.utils.plane_pcd_utils import *
from mirror3d.annotation.plane_annotation.plane_annotation_tool import *


def gen_color_depth_mask(mask_main_folder ="", raw_d_main_folder ="", color_main_folder =""):
    mask_list = [i.strip() for i in os.popen("find -L {} -type f".format(mask_main_folder)).readlines()]  
    raw_d_list = [i.strip() for i in os.popen("find -L {} -type f".format(raw_d_main_folder)).readlines()]
    color_list = [i.strip() for i in os.popen("find -L {} -type f".format(color_main_folder)).readlines()]


 if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--data_main_folder', default="")
    parser.add_argument(
        '--mask_main_folder', default="")
    parser.add_argument(
        '--rawD_main_folder', default="")
    parser.add_argument(
        '--color_main_folder', default="")
    args = parser.parse_args()
