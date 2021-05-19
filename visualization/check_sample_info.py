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


def check_mirror_normal_in_img_info_Json(data_root_path, json_path, f):
    import open3d as o3d
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())
    raw_image_path = "" # TODO
    depth_image_path = "" # TODO
    pcd = get_pcd_from_rgbd_depthPath(f, depth_image_path, raw_image_path)
    mirror_normal = np.array(anno_info["mirror_normal_camera"])
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



def check_mirror_normal_in_cocoJson(data_root_path, json_path, f):
    import open3d as o3d
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())
    for item in anno_info["annotations"]:
        raw_image_path = os.path.join(data_root_path, item["image_path"])
        depth_image_path = os.path.join(data_root_path, item["depth_path"])
        mask_image_path = os.path.join(data_root_path, item["mask_path"])
        print(raw_image_path, mask_image_path)
        pcd = get_pcd_from_rgbd_depthPath(f, depth_image_path, raw_image_path)
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
        data_root_path = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu"
        json_path = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json"
        f = 519
        check_mirror_normal_in_cocoJson(data_root_path, json_path, f)