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





def gen_color_depth_mask(data_main_folder = "", mask_main_folder = "", rawD_main_folder = "", color_main_folder = ""):
    
    mask_list = [i.strip() for i in os.popen("find -L {} -type f".format(mask_main_folder)).readlines()]  
    rawD_list = [i.strip() for i in os.popen("find -L {} -type f".format(rawD_main_folder)).readlines()]  
    color_list = [i.strip() for i in os.popen("find -L {} -type f".format(color_main_folder)).readlines()]  
    


    pass


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
