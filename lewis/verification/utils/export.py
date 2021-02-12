import cv2
from skimage import io
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import multiprocessing

def get_pcd_from_rgbd(f, depth_img_path, color_img_path, mask_path, save_path):
    mask  = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
    if mask_path is not None and len(mask.shape)>2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    d = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img/255

    h, w = d.shape
    x_cam = []
    y_cam = []
    z_cam = []
    colors = []
    xyz = []

    for y in range(h):
        for x in range(w):
            if  mask_path is not None and mask[y][x] > 0:
                colors.append([0,0,1])
                xyz.append([(x - w/2) * (d[y][x]/f),(y - h/2) * (d[y][x]/f),d[y][x]])
            else:
                colors.append(color_img[y][x])
                xyz.append([(x - w/2) * (d[y][x]/f),(y - h/2) * (d[y][x]/f),d[y][x]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz,axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors,axis=0))
    o3d.io.write_point_cloud(save_path, pcd)
    print(save_path)

if __name__ == "__main__":
    f = 1076
    clamp_depth_folder = "/Volumes/Data/paper/Rebuttal/m3d_verify/new_clamped_depth"
    raw_folder = "/Volumes/Data/paper/Rebuttal/m3d_verify/raw"
    mask_folder = "/Volumes/Data/paper/Rebuttal/m3d_verify/instance_mask"
    pcd_save_folder = "/Volumes/Data/paper/Rebuttal/m3d_verify/pcd"

    pool = multiprocessing.Pool(processes=8)
    for depth in tqdm(os.listdir(clamp_depth_folder)):
        if depth.startswith('.'):
            continue
        depth_img_path = os.path.join(clamp_depth_folder, depth)
        color_img_path = os.path.join(raw_folder, depth.replace("_d","_i"))
        mask_img_path = os.path.join(mask_folder, depth.replace("_d","_i"))
        save_path = os.path.join(pcd_save_folder, color_img_path.split("/")[-1].split(".")[0]+".ply")
        pool.apply_async(get_pcd_from_rgbd, (f, depth_img_path, color_img_path, mask_img_path, save_path, ))
    pool.close()
    pool.join()