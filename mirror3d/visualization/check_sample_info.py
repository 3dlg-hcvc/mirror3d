import numpy as np
import cv2
import json
import argparse
import open3d as o3d
from mirror3d.utils.algorithm import *
from mirror3d.utils.general_utils import *
from mirror3d.utils.plane_pcd_utils import *
from mirror3d.annotation.plane_annotation.plane_annotation_tool import *


def check_mirror_normal_in_one_json(color_img_path, mask_img_path, depth_img_path, json_path, f):
    with open(json_path, 'r') as j:
        anno_info = json.loads(j.read())

    for item in anno_info:
        instance_index = item["mask_id"]
        mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
        binary_instance_mask = (mask == instance_index)
        pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask,
                                          color=[0, 0.2, 0.6])
        mirror_normal = np.array(item["normal"])

        mirror_points = get_points_in_mask(f, depth_img_path, mirror_mask=binary_instance_mask)
        mirror_pcd = o3d.geometry.PointCloud()
        mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0))
        mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0)))
        mirror_plane = get_mirror_init_plane_from_mirrorbbox(item["plane"], mirror_bbox)
        plane_center = np.mean(np.array(mirror_plane.vertices), axis=0)
        # get mirror normal 
        if "mp3d" in json_path:
            vec_len = 2
        else:
            vec_len = 0.5
        ratio = np.abs(1000 / mirror_normal[0])
        begin = plane_center
        end = [begin[0] + mirror_normal[0] * ratio, begin[1] + mirror_normal[1] * ratio,
               begin[2] + mirror_normal[2] * ratio]
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(mirror_plane)
        vis.add_geometry(get_mesh_by_start_end(begin, end, vec_len=vec_len))
        vis.get_render_option().point_size = 1.0
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--color_img_path', default="")
    parser.add_argument(
        '--mask_img_path', default="")
    parser.add_argument(
        '--depth_img_path', default="")
    parser.add_argument(
        '--json_path', default="")
    parser.add_argument(
        '--f', default=519, type=int,
        help="camera focal length; tips : 1074 for Matterport3D, 519 for NYUv2-small, 574 for ScanNet")
    args = parser.parse_args()
    check_mirror_normal_in_one_json(args.color_img_path, args.mask_img_path, args.depth_img_path, args.json_path,
                                    args.f)
