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



if __name__ == "__main__":


    # TODO add option here

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--data_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
    parser.add_argument(
        '--index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    args = parser.parse_args()
    
    # plane_anno_tool = Plane_annotation_tool(args.data_main_folder, args.index, False)
    # plane_anno_tool.anno_update_depth_from_imgInfo()
    plane_anno_tool = Data_post_processing(args.data_main_folder, args.index, False)
    plane_anno_tool.data_clamping()