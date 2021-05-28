import numpy as np
import cv2
import argparse
import os
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
import json
from annotation.plane_annotation.plane_annotation_tool import *
from PIL import ImageColor
import open3d as o3d



def check_mirror_normal_in_one_json(data_root_path, json_path, f, mask_version):
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())

    color_img_path = json_path.replace("mirror_plane","mirror_color_images").replace("json","jpg")
    mask_path =  json_path.replace("mirror_plane","mirror_instance_mask_{}".format(mask_version)).replace("json","png")
    if "mp3d" in color_img_path:
        depth_img_path = json_path.replace("mirror_plane","refined_meshD_{}".format(mask_version)).replace("json","png")
        depth_img_path = rreplace(depth_img_path,"i","d")
    else:
        depth_img_path = json_path.replace("mirror_plane","refined_sensorD_{}".format(mask_version)).replace("json","png")
    print("instance mask path : ", mask_path)
    print("color_img_path : ", color_img_path)
    print("depth_img_path : ", depth_img_path)
    for item in anno_info:
        instance_index = item["mask_id"]
        instance_index = ImageColor.getcolor("#{}".format(instance_index), "RGB")
        mask = cv2.imread(mask_path)
        binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
        pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask, color=[0,0.2,0.6])
        mirror_normal = np.array(item["normal"])

        mirror_points = get_points_in_mask(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)
        mirror_pcd = o3d.geometry.PointCloud()
        mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
        mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
        mirror_plane = get_mirror_init_plane_from_mirrorbbox(item["plane"], mirror_bbox)
        plane_center = np.mean(np.array(mirror_plane.vertices), axis=0)
        # get mirror normal 
        if "mp3d" in json_path:
            vec_len = 2
        else:
            vec_len = 0.5
        ratio = np.abs(1000 / mirror_normal[0])
        begin = plane_center
        end = [begin[0] + mirror_normal[0]*ratio, begin[1] + mirror_normal[1]*ratio, begin[2] + mirror_normal[2]*ratio]
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(mirror_plane)
        vis.add_geometry(get_mesh_by_start_end(begin,end,vec_len=vec_len))
        vis.get_render_option().point_size = 1.0
        vis.run()
        vis.destroy_window()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--data_root_path', default="")
    parser.add_argument(
        '--json_path', default="")
    parser.add_argument(
    '--f', default=519, type=int, help="camera focal length; tips : 1074 for Matterport3D, 519 for NYUv2-small, 574 for ScanNet")
    parser.add_argument(
    '--mask_version', default="precise", help="2 mask version : precise/ coarse")
    args = parser.parse_args()

    check_mirror_normal_in_one_json(args.data_root_path, args.json_path, args.f, args.mask_version)