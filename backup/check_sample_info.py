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


def check_mirror_normal_in_one_json(data_root_path, json_path, f, mask_version):
    import open3d as o3d
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())

    color_img_path = json_path.replace("mirror_plane","mirror_color_images").replace("json","jpg")
    mask_img_path =  json_path.replace("mirror_plane","mirror_instance_mask_{}".format(mask_version)).replace("json","png")
    if "m3d" in color_img_path:
        depth_img_path = json_path.replace("mirror_plane","refined_meshD_{}".format(mask_version)).replace("json","png")
        depth_img_path = rreplace(depth_img_path,"i","d")
    else:
        depth_img_path = json_path.replace("mirror_plane","refined_sensorD_{}".format(mask_version)).replace("json","png")
    
    pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path)
    for item in anno_info.items():
        index = item[0]
        instance_index = [int(i) for i in index.split("_")]
        mask = cv2.imread(mask_img_path)
        binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
        mirror_normal = np.array(item[1]["mirror_normal"])


        mirror_points = get_points_in_mask(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)
        mirror_pcd = o3d.geometry.PointCloud()
        mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
        mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
        mirror_plane = get_mirror_init_plane_from_mirrorbbox(item[1]["plane_parameter"], mirror_bbox)
        plane_center = np.mean(np.array(mirror_plane.vertices), axis=0)
        # get mirror normal 
        ratio = 1000 / mirror_normal[0]
        p1 = plane_center
        p2 = [p1[0] + mirror_normal[0]*ratio, p1[1] + mirror_normal[1]*ratio, p1[2] + mirror_normal[2]*ratio]
        points = [
            p1,
            p2
            ]
        mirror_normal_line = [[0,1]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(mirror_normal_line),
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(mirror_plane)
        vis.add_geometry(line_set)
        vis.get_render_option().point_size = 1.0
        vis.run()
        vis.destroy_window()
        



def check_mirror_normal_in_cocoJson(data_root_path, json_path, f):
    import open3d as o3d
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())
    for item in anno_info["annotations"]:
        color_img_path = os.path.join(data_root_path, item["image_path"])
        depth_img_path = os.path.join(data_root_path, item["mesh_refined_path"])
        mask_image_path = os.path.join(data_root_path, item["mask_path"])
        print(color_img_path, mask_image_path)
        pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path)
        mirror_normal = np.array(item["mirror_normal_camera"])
        p1 = [0, 0, 0]
        p2 = list(mirror_normal*1000)
        points = [
            p1,
            p2
            ]
        mirror_normal_line = [[0,1]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(mirror_normal_line),
        )


        o3d.visualization.draw_geometries([line_set, pcd])

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Get Setting :D')
        parser.add_argument(
            '--stage', default="2")
        parser.add_argument(
            '--data_root_path', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu")
        parser.add_argument(
            '--json_path', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/img_info/1003.json")
        parser.add_argument(
        '--f', default=519, type=int, help="camera focal length")
        args = parser.parse_args()

        if args.stage == "1":
            check_mirror_normal_in_cocoJson(args.data_root_path, args.json_path, args.f)
        elif args.stage == "2":
            check_mirror_normal_in_one_json(args.data_root_path, args.json_path, args.f)