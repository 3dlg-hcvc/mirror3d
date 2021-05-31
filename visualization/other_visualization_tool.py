import open3d as o3d
import os
import numpy as np
from utils.general_utils import *
from utils.plane_pcd_utils import *
import cv2
import argparse

# ---------------------------------------------------------------------------- #
#                function to iteratively visualize a point cloud               #
# ---------------------------------------------------------------------------- #
def visualize_pcd():
    while 1:
        input_path = input("(q : Quit) cloud point path: ")
        if input_path == "q" :
            exit(1)
        pcd = o3d.io.read_point_cloud(input_path)
        xyz = np.array(pcd.points)
        X = xyz[:,0]
        Y = xyz[:,1]
        Z = xyz[:,2]

        print("X max {:.2f} X min {:.2f} X gap {:.2f}".format(X.max(), X.min(), X.max()-X.min()))
        print("Y max {:.2f} Y min {:.2f} Y gap {:.2f}".format(Y.max(), Y.min(), Y.max()-Y.min()))
        print("Z max {:.2f} Z min {:.2f} Z gap {:.2f}".format(Z.max(), Z.min(), Z.max()-Z.min()))

        o3d.visualization.draw_geometries([pcd])


def vislize_pcd_from_rgbd(depth_img_path, color_img_path, f, save_folder=""):

    pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path)
    
    if os.path.exists(save_folder):
        pcd_save_path = os.path.join(save_folder, "{}.ply".format(color_img_path.split("/")[-1].split(".")[0]))
        o3d.io.write_point_cloud(pcd_save_path, pcd)
        print("pcd saved to {}".format(pcd_save_path))
    else:
        print("visulizing {}".format(color_img_path))
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument(
        '--depth_img_path', default="")
    parser.add_argument(
        '--color_img_path', default="")
    parser.add_argument(
        '--img_info_path', default="")
    parser.add_argument(
        '--instance_index', default="")
    parser.add_argument(
        '--f', default=1075, type=int, help="camera focal length")

    args = parser.parse_args()

    if args.stage == "1":
        visualize_pcd()
    elif args.stage == "2":
        vislize_pcd_from_rgbd(args.depth_img_path, args.color_img_path, args.f, "")

