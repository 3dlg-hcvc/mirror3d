import cv2
from skimage import io
import open3d as o3d
import numpy as np
import os
import argparse
import sys
sys.path.append("../")
from utils.file_io import *

# TODO: modify the variable name
def get_progress(progress_save_folder, pcd_save_folder):
    pcd_path_list = []
    for pcd_name in os.listdir(pcd_save_folder):
        if pcd_name.startswith('.'):
            continue
        pcd_path_list.append(os.path.join(pcd_save_folder, pcd_name))


    error_txt = os.path.join(progress_save_folder, "error_list.txt")
    correct_txt = os.path.join(progress_save_folder, "correct_list.txt")

    if os.path.exists(error_txt):
        error_list = read_txt(error_txt)
    else:
        error_list = []

    if os.path.exists(correct_txt):
        correct_list = read_txt(correct_txt)
    else:
        correct_list = []

    path_to_annotate = []
    annotated_paths = []
    path_to_annotate = list(set(pcd_path_list) - set(error_list) - set(correct_list))
    annotated_paths = list(set(pcd_path_list) - set(path_to_annotate))
    if len(path_to_annotate) > 0:
        path_to_annotate.sort()
    if len(annotated_paths) > 0:
        annotated_paths.sort()
    return annotated_paths, path_to_annotate, error_list, correct_list


def update_verification_progress(progress_save_folder, error_list, correct_list):
    error_txt_path = os.path.join(progress_save_folder, "error_list.txt")
    correct_txt_path = os.path.join(progress_save_folder, "correct_list.txt")
    save_txt(error_txt_path, error_list)
    save_txt(correct_txt_path, correct_list)

if __name__ == "__main__":
    f = 1076
    clamp_depth_folder = "/Volumes/Data/paper/Rebuttal/clamp_issue/ref_depth"
    raw_folder = "/Volumes/Data/paper/Rebuttal/clamp_issue/raw"
    mask_folder = "/Volumes/Data/paper/Rebuttal/clamp_issue/mask"

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
    '--pcd_save_folder', default="/local-scratch/share_data/matterport_extracted/400_incomplete_mirror/mirror_refine/point_cloud", type=str,
    help="folder path to save the output initial point cloud (ransac mirror area -- green points; original mirror area --- red points)")
    
    parser.add_argument(
    '--verification_progress_save_folder', default="/local-scratch/share_data/matterport_extracted/400_incomplete_mirror/mirror_refine/progress", type=str, 
    help="folder contains current annotation progress information")

    args = parser.parse_args()
    
    verified_paths, path_to_verify , error_list, correct_list = get_progress(args.verification_progress_save_folder,args.pcd_save_folder)
    
    while 1:
        if len(path_to_verify) == 0:
            print("verification finished ! XD")
            break
        current_pcd_path = path_to_verify.pop()
        pcd = o3d.io.read_point_cloud(current_pcd_path)
        print(current_pcd_path)
        coor_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8000,  origin=[0,0,0])      
        o3d.visualization.draw_geometries([pcd, coor_ori])
        option = input("option :\n\
            (1) t : current plane parameter is correct\n\
            (2) f : current plane parameter is wrong\n\
            (3) exit : save the result and exit\n")
        if option == "t":
            correct_list.append(current_pcd_path)
            update_verification_progress(args.verification_progress_save_folder, error_list, correct_list)
            verified_paths, path_to_verify, error_list, correct_list = get_progress(args.verification_progress_save_folder,args.pcd_save_folder)
        elif option == "f":
            error_list.append(current_pcd_path)
            update_verification_progress(args.verification_progress_save_folder, error_list, correct_list)
            verified_paths, path_to_verify, error_list, correct_list = get_progress(args.verification_progress_save_folder,args.pcd_save_folder)
        elif option == "exit":
            update_verification_progress(args.verification_progress_save_folder, error_list, correct_list)
            verified_paths, path_to_verify , error_list, correct_list = get_progress(args.verification_progress_save_folder,args.pcd_save_folder)
            print("current progress {} / {}".format(len(verified_paths), len(verified_paths) + len(path_to_verify)))
            exit(1)
        else:
            print("invalid input, please input again :D")
            verified_paths, path_to_verify, error_list, correct_list = get_progress(args.verification_progress_save_folder,args.pcd_save_folder)
            continue

