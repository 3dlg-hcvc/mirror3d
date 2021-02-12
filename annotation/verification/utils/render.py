import open3d as o3d
import os
import json
import time
import cv2
import numpy as np
from file_io import *

def read_pcd_from_path(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

if __name__ == "__main__":
    err_path = "/Volumes/Data/paper/Rebuttal/m3d_verify/progress/error_list.txt"
    err_list = read_txt(err_path)
    for path in err_list:
        print(path)
        pcd = read_pcd_from_path(path)
        coor_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8000,  origin=[0,0,0])      
        o3d.visualization.draw_geometries([pcd, coor_ori])
