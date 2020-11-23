import cv2
from skimage import io
import open3d as o3d
import numpy as np
import os
import json
import PIL
from PIL import Image
from tqdm import tqdm
from operator import add
import argparse
import multiprocessing
import math


def read_json(json_path):
    with open(json_path, 'r') as j:
        info  = json.loads(j.read())
    return info

def read_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def get_coord_transform_mat(plane_param):
    a, b, c, d = plane_param
    N = np.array([a, b, c])
    # find 3 non-collinear points
    A = np.array([0, 0, -d/c])
    B = np.array([0, 1, -(b+d)/c])
    AB = np.subtract(A, B)
    BC = np.cross(AB, N)
    C = np.add(B, BC)

    # get 3 normalized vectors
    U  = AB/np.linalg.norm(AB)
    V  = BC/np.linalg.norm(BC)
    uN = N/np.linalg.norm(N)

    u = np.add(A, U)
    v = np.add(A, V)
    n = np.add(A, uN)

    D = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1]])
    S = np.vstack( [np.stack([A, u, v, n], axis=1), np.array([1,1,1,1])] )
    inv_S = np.linalg.inv(S)
    M = np.matmul(D, inv_S)    
    return M

def get_project_coord(xyz, plane_param):
    a, b, c, d = plane_param

    n = np.array([a, b, c])
    V0 = np.array([0, 0, -d/c])
    P0 = np.array([0,0,0])
    P1 = np.array(xyz)

    j = P0 - V0
    u = P1-P0
    N = -np.dot(n,j)
    D = np.dot(n,u)
    sI = N / D
    I = P0+ sI*u
    return list(I)

def get_2d_bbox(on_plane_xy):
    max_x, max_y = np.amax(on_plane_xy, axis=0)
    min_x, min_y = np.amin(on_plane_xy, axis=0)
    return [min_x, max_x, min_y, max_y]

def get_plane_mesh(M, xxyy, camera_normal, instance_idx):
    min_x, max_x, min_y, max_y = xxyy
    inv_M = np.linalg.inv(M)
    p1 = np.matmul(inv_M, np.array([min_x, min_y, 0, 1]).T)[:3]
    p2 = np.matmul(inv_M, np.array([min_x, max_y, 0, 1]).T)[:3]
    p3 = np.matmul(inv_M, np.array([max_x, min_y, 0, 1]).T)[:3]
    p4 = np.matmul(inv_M, np.array([max_x, max_y, 0, 1]).T)[:3]
    mirror_plane = o3d.geometry.TriangleMesh()
    mirror_plane.vertices = o3d.utility.Vector3dVector(np.array([p1, p2, p3, p4]))
    mirror_plane.triangles= o3d.utility.Vector3iVector(np.array([[0,1,2],[2,1,3]]))    
    mirror_plane.paint_uniform_color([0.08*instance_idx, 0.08*instance_idx, 1])
    mirror_plane.compute_triangle_normals()
    plane_normal = np.asarray(mirror_plane.triangle_normals)[0]
    # check if it is visible from the camera view
    if np.dot(plane_normal, camera_normal) > 0:        
        mirror_plane.triangles= o3d.utility.Vector3iVector(np.array([[0,2,1],[2,3,1]]))
    return mirror_plane

def get_pcd(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz,axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors,axis=0))
    return pcd

def get_point_plane_distance(xyz, plane_param):
    x, y, z = xyz
    a, b, c, d = plane_param
    d = abs(a*x+b*y+c*z+d) / math.sqrt(a**2+b**2+c**2)
    return d

def export_from_rgbd(f, color_img_path, mask_path, sens_depth_img_path, img_info_path, shift, args):
    print("color path: "+color_img_path)
    print("sensor depth path: "+sens_depth_img_path)
    print("mask path: "+mask_path)
    # Read plane info and images
    img_info = read_json(img_info_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
    sens_d = cv2.imread(sens_depth_img_path, cv2.IMREAD_ANYDEPTH)
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img/255

    h, w = sens_d.shape
    mirror_ref_colors, mirror_sens_colors, rest_colors = [], [], []
    mirror_ref_xyz, mirror_sens_xyz, rest_xyz = [], [], []
    mirror_plane_mesh = []

    # Draw pcd and mesh mask by mask
    for idx, instance_index in enumerate(np.unique(mask)):
        on_plane_xy = []

        if instance_index == 0: # background
            continue
        plane_param = img_info[str(instance_index)]["plane_parameter"]
        M = get_coord_transform_mat(plane_param)

        current_instance_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        current_instance_mask[current_instance_mask!= instance_index] = 0
        mirror_border_mask = cv2.dilate(current_instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))) - current_instance_mask

        for y in range(h):
            for x in range(w):
                if  current_instance_mask[y][x] > 0:                    
                    mirror_color = color_img[y][x]
                    mirror_sens_colors.append(mirror_color)
                    mirror_sens_xyz.append([(x - w/2) * (sens_d[y][x]/f),(y - h/2) * (sens_d[y][x]/f),sens_d[y][x]])
                    # Project 3d points on the mirror plane
                    proj_xyz = get_project_coord([(x - w/2), (y - h/2), f], plane_param)        
                    mirror_ref_xyz.append(proj_xyz)
                    mirror_ref_colors.append(mirror_color)
                    # Save the 2d coordinates on the plane    
                    on_plane_xy.append(np.matmul(M, np.transpose(np.append(proj_xyz,1)))[:2])
                elif mirror_border_mask[y][x] > 0:
                    clamp_xyz = get_project_coord([(x - w/2), (y - h/2), f], plane_param)                    
                    
                    # ScanNet has 0 depth value on the image border, should be ignored
                    if sens_d[y][x]!=0:  
                        # Refined mask should preserve the occlusion 
                        if args.is_refined:                        
                            rest_colors.append(color_img[y][x])
                            # Check if it is within 20cm from the front of mirror
                            if get_point_plane_distance(clamp_xyz, plane_param) < 200*shift:
                                rest_xyz.append(clamp_xyz)
                            else:
                                rest_xyz.append([(x - w/2) * (sens_d[y][x]/f),(y - h/2) * (sens_d[y][x]/f),sens_d[y][x]])
                        else:
                            rest_colors.append(color_img[y][x])
                            rest_xyz.append(clamp_xyz)

        # Accumulatively collect each plane mesh, idx is to paint different instance differently
        on_plane_xy = np.asarray(on_plane_xy)
        xxyy = get_2d_bbox(on_plane_xy)
        if mirror_plane_mesh == []:
            mirror_plane_mesh = get_plane_mesh(M, xxyy, np.array([0,0,f]), idx)
        else:
            mirror_plane_mesh += get_plane_mesh(M, xxyy, np.array([0,0,f]), idx)
    
    mask[mask!=0] = 1
    mirror_border_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))) - mask

    # Create non-mirror region point cloud    
    for y in range(h):
        for x in range(w):
            # Avoid adding clamped points again
            if mirror_border_mask[y][x] > 0:
                continue
            elif mask[y][x] == 0:
                xyz = [(x - w/2) * (sens_d[y][x]/f),(y - h/2) * (sens_d[y][x]/f),sens_d[y][x]]
                rest_colors.append(color_img[y][x])
                rest_xyz.append(xyz)        

    # Generate point cloud
    mirror_sens_pcd = get_pcd(mirror_sens_xyz, mirror_sens_colors)
    mirror_ref_pcd = get_pcd(mirror_ref_xyz, mirror_ref_colors)
    rest_pcd = get_pcd(rest_xyz, rest_colors)

    return mirror_ref_pcd, mirror_sens_pcd, rest_pcd, mirror_plane_mesh

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Get Setting')
    parser.add_argument('--f', default=1076, type=int)
    parser.add_argument('--dataset', default="", type=str)
    parser.add_argument('--is_refined', type=bool)
    parser.add_argument('--color_img_folder', default="", type=str)
    parser.add_argument('--mask_img_folder', default="", type=str)
    parser.add_argument('--sens_depth_img_folder', default="", type=str)
    parser.add_argument('--img_info_folder', default="", type=str)
    parser.add_argument('--export_folder', default="", type=str)
    args = parser.parse_args()

    # pool = multiprocessing.Pool(processes=16) 
    for mask_key in os.listdir(args.mask_img_folder):
        if mask_key.startswith('.'):
            continue
        # Load image paths
        mask_img_path = os.path.join(args.mask_img_folder, mask_key)
        img_info_path = os.path.join(args.img_info_folder,  mask_key.replace(".png",".json"))        

        # Matterport3D has a different id patterns for color and depth image
        if args.dataset == "m3d":
            color_img_path = os.path.join(args.color_img_folder, mask_key.replace(".png",".jpg"))
            depth_key = "{}_{}_{}".format(mask_key.split("_")[0], mask_key.split("_")[1].replace("i","d"), mask_key.split("_")[2])
            sens_depth_img_path = os.path.join(args.sens_depth_img_folder, depth_key)
        else:
            color_img_path = os.path.join(args.color_img_folder, mask_key)
            sens_depth_img_path  = os.path.join(args.sens_depth_img_folder, mask_key)        
        
        if args.dataset == "scannet" or args.dataset == "nyu":
            shift = 1
        elif args.dataset == "m3d":
            shift = 4

        # Export pcd and mesh
        outputs = export_from_rgbd(args.f, color_img_path, mask_img_path, sens_depth_img_path, img_info_path, shift, args)
        
        # Make directories
        save_folder = os.path.join(args.export_folder, mask_key.split(".")[0])
        os.makedirs(save_folder, exist_ok=True)
        
        # Write files
        output_name = ["mirror_ref_pcd.ply", "mirror_sens_pcd.ply", "rest_pcd.ply", "mirror_mesh.ply"]
        for idx, file_name in enumerate(output_name):
            save_path = os.path.join(save_folder,file_name)
            o3d.io.write_point_cloud(save_path, outputs[idx])
