import open3d as o3d
import os
import numpy as np
from utils.plane_pcd_utils import *
import cv2

# ---------------------------------------------------------------------------- #
#                function to iteratively visualize a point cloud               #
# ---------------------------------------------------------------------------- #
def visualize_single_image():
    while 1:
        input_path = input("cloud point path : ")
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
    
    visualize_single_image()
    depth_img_path = "/local-scratch/jiaqit/exp/reannotate/hole_refined_depth/718.png"
    f = 519
    color_img_path = "/local-scratch/jiaqit/exp/reannotate/raw/718.png"
    pcd_save_folder = "/local-scratch/jiaqit/exp/reannotate/vis"
    #vislize_pcd_from_rgbd(depth_img_path, color_img_path, f, pcd_save_folder)
    
