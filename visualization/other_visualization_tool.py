import open3d as o3d
import os
import numpy as np
from utils.general_utlis import *
from utils.plane_pcd_utils import *
import cv2
import argparse

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


def check_one_sample(color_img_path, depth_img_path, img_info_path, instance_id, f):
    pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path)
    img_info = read_json(img_info_path)
    plane_parameter = img_info[instance_id]["plane_parameter"]
    mask_path = color_img_path.replace("raw", "instance_mask")
    instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),[ int(i) for i in instance_id.split("_")])
    mirror_points = get_points_in_mask(f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
    mirror_pcd = o3d.geometry.PointCloud()
    mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
    mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))) 
    mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox)
    o3d.visualization.draw_geometries([pcd, mirror_plane])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="3")
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
        visualize_single_image()
    elif args.stage == "2":
        vislize_pcd_from_rgbd(args.depth_img_path, args.color_img_path, args.f, "")
    elif args.stage == "3":
        check_one_sample(args.color_img_path, args.depth_img_path, args.img_info_path, args.instance_index, args.f)

