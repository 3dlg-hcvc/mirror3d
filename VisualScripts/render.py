import open3d as o3d
import os
import json
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info

def read_pcd_from_path(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def read_mesh_from_path(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh

def export_transparent_mask(mask_path, overlay_save_path):
    mask = cv2.imread(mask_img_path, -1)
    b, g, r = cv2.split(mask)
    a = np.ones(b.shape, dtype=b.dtype) * 127
    mask = cv2.merge((b, g, r, a))
    mask[np.where((mask == [0,0,0,127]).all(axis = 2))] = [0,0,0,0]

    cv2.imwrite(overlay_save_path, mask)

def save_view_by_json(pcd, json_file_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    view_list = read_json(json_file_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for index, one_view_param in enumerate(view_list):
        ss_save_name = os.path.join(save_folder, "screenshot_{}.png".format(index))
        # Visualize Point Cloud
        for item in pcd:
            vis.add_geometry(item)
        ctr = vis.get_view_control()
        vis.get_view_control().set_front(one_view_param["trajectory"][0]["front"])
        vis.get_view_control().set_lookat(one_view_param["trajectory"][0]["lookat"])
        vis.get_view_control().set_zoom(one_view_param["trajectory"][0]["zoom"])
        vis.get_view_control().set_up(one_view_param["trajectory"][0]["up"])

        # Updates
        for item in pcd:
            vis.update_geometry(item)
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(1)
        vis.capture_screen_image(ss_save_name)
        print("image saved to :", ss_save_name)
        # Remove previous geometry
        for item in pcd:
            vis.remove_geometry(item)

    # Close outside the loop        
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting')
    parser.add_argument('--color_img_folder', default="", type=str)
    parser.add_argument('--mask_img_folder', default="", type=str)
    parser.add_argument('--ply_folder', default="", type=str)
    parser.add_argument('--save_folder', default="", type=str)
    parser.add_argument('--mode', default="view", type=str)
    parser.add_argument('--json_file_path', default="/Users/lewislin/Desktop/proj-mirror-annotation/Mirror3D/config/view_list.json", type=str)

    args = parser.parse_args()
    
    if args.mode == "view":
        for key in os.listdir(args.ply_folder):
            # ignore the hidden files
            if key.startswith('.'):
                continue
            ply_folder = os.path.join(args.ply_folder, key)
            rest_pcd = read_pcd_from_path(os.path.join(ply_folder, "rest_pcd.ply"))
            mirror_sens_pcd = read_pcd_from_path(os.path.join(ply_folder, "mirror_sens_pcd.ply"))
            mirror_mesh = read_mesh_from_path(os.path.join(ply_folder,"mirror_mesh.ply"))
            o3d.visualization.draw_geometries([rest_pcd, mirror_sens_pcd, mirror_mesh])

    if args.mode == "screenshot":
        for key in os.listdir(args.ply_folder):
            # ignore the hidden files
            if key.startswith('.'):
                continue
            ply_folder = os.path.join(args.ply_folder, key)
            rest_pcd = read_pcd_from_path(os.path.join(ply_folder, "rest_pcd.ply"))
            mirror_sens_pcd = read_pcd_from_path(os.path.join(ply_folder, "mirror_sens_pcd.ply"))
            mirror_mesh = read_mesh_from_path(os.path.join(ply_folder,"mirror_mesh.ply"))
            save_path = os.path.join(args.save_folder, key)
            save_view_by_json([rest_pcd, mirror_sens_pcd, mirror_mesh], args.json_file_path, save_path)

    if args.mode == "transparent_mask":
        os.makedirs(args.save_folder, exist_ok=True)
        for mask_key in os.listdir(args.mask_img_folder):            
            mask_img_path = os.path.join(args.mask_img_folder, mask_key)
            color_img_path = os.path.join(args.color_img_folder, mask_key.replace(".png",".jpg"))
            save_path = os.path.join(args.save_folder, mask_key)
            export_transparent_mask(mask_img_path, save_path)

