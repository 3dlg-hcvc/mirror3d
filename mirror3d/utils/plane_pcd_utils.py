import numpy as np
import cv2
from mirror3d.utils.algorithm import *
from mirror3d.utils.general_utils import *
from sympy import *
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------- #
#                   get vector from begin to end                               #
# ---------------------------------------------------------------------------- #


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    q_cross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return q_cross_prod_mat


def calculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_arr, pVec_Arr) == -1:
        q_trans_mat = -np.eye(3, 3)
    elif np.dot(z_unit_arr, pVec_Arr) == 1:
        q_trans_mat = np.eye(3, 3)
    else:
        q_trans_mat = np.eye(3, 3) + z_c_vec_mat + \
                      np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_arr, pVec_Arr))
    q_trans_mat *= scale
    return q_trans_mat


def get_mesh_by_start_end(begin, end, color=[0.6, 0.6, 1], vec_len=1):
    import open3d as o3d
    vec_arr = np.array(end) - np.array(begin)
    rot_mat = np.float32(calculate_align_mat(vec_arr))
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cone_height=0.2 * vec_len, cone_radius=0.08 * vec_len,
                                                        cylinder_height=0.8 * vec_len, cylinder_radius=0.02 * vec_len)
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.rotate(rot_mat, center=(0, 0, 0))
    mesh_arrow.translate(np.array(begin))
    return mesh_arrow


# ---------------------------------------------------------------------------- #
#                   get grayscale mask from 3 channel's mask                   #
# ---------------------------------------------------------------------------- #
def get_grayscale_instanceMask(mask, instance_index):
    current_instance_mask = np.zeros(mask.shape[:-1])
    h, w, _ = mask.shape
    for i in range(h):
        for j in range(w):
            if (mask[i][j] == instance_index).all():
                current_instance_mask[i][j] = 1
            else:
                current_instance_mask[i][j] = 0
    return current_instance_mask


# ---------------------------------------------------------------------------- #
#                   get_mirror_parameter_from_xyzs_by_ransac                   #
# ---------------------------------------------------------------------------- #
def get_mirror_parameter_from_xyzs_by_ransac(xyzs):
    def augment(xyzs):
        axyz = np.ones((len(xyzs), 4))
        axyz[:, :3] = xyzs
        return axyz

    def estimate(xyzs):
        axyz = augment(xyzs[:3])
        return np.linalg.svd(axyz)[-1][-1, :]

    def is_inlier(coeffs, xyz, threshold):
        return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

    max_iterations = 200
    goal_inliers = len(xyzs) * 0.9
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    return m


# ---------------------------------------------------------------------------- #
#                 get_pcd_mirror_points_from_rgbd_and_intrinsic                #
# ---------------------------------------------------------------------------- #
def get_pcd_mirror_points_from_rgbd_and_intrinsic(intrinsic_matrix, depth_shift, depth_img_path, color_img_path,
                                                  mirror_mask=None, color=None):
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]

    cx = 640
    cy = 512

    import open3d as o3d
    d = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH) / depth_shift
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img / 255

    h, w = d.shape
    x_cam = []
    y_cam = []
    z_cam = []
    colors = []
    xyz = []
    mirror_xyz = []
    mirror_colors = []

    for y in range(h):
        for x in range(w):
            if mirror_mask is not None and mirror_mask[y][x]:
                colors.append(color_img[y][x])
                xyz.append([(x - cx) * (d[y][x] / fx), (y - cy) * (d[y][x] / fy), d[y][x]])
                if color is not None:
                    mirror_colors.append(color)
                else:
                    mirror_colors.append(color_img[y][x])
                mirror_xyz.append([(x - cx) * (d[y][x] / fx), (y - cy) * (d[y][x] / fy), d[y][x]])
            else:
                colors.append(color_img[y][x])
                xyz.append([(x - cx) * (d[y][x] / fx), (y - cy) * (d[y][x] / fy), d[y][x]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors, axis=0))

    mirror_pcd = o3d.geometry.PointCloud()
    mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_xyz, axis=0))
    mirror_pcd.colors = o3d.utility.Vector3dVector(np.stack(mirror_colors, axis=0))

    return pcd, mirror_pcd


# --------------------------------------------------------------------------------------------------------------------------------  #
# get_colored_pcd | light blue points : points generated based on plane_parameter | blue points : colored point cloud's mirror area #
# --------------------------------------------------------------------------------------------------------------------------------  #
def get_colored_pcd(f=538, depth_img_path="", color_img_path="", mirror_mask=None, mirror_plane_mask=None,
                    plane_parameter=[1, 1, 1, 1]):
    import open3d as o3d
    d = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img / 255

    h, w = d.shape
    x_cam = []
    y_cam = []
    z_cam = []
    colors = []
    xyz = []

    for y in range(h):
        for x in range(w):
            if mirror_mask is not None and mirror_mask[y][x]:
                colors.append([0, 0, 1])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])
            else:
                colors.append(color_img[y][x])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])

    a, b, c, d = plane_parameter
    for y in range(h):
        for x in range(w):
            if mirror_plane_mask[y][x]:
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u
                xyz.append(I)
                colors.append([0.1, 1, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors, axis=0))
    return pcd

def get_z_from_plane(plane_parameter, x, y):
    [a, b, c, d] = plane_parameter
    z = (-d - a * x - b * y) / c
    return z

# ---------------------------------------------------------------------------- #
#                    get_mirror_init_plane based on 3 points                   #
# ---------------------------------------------------------------------------- #
def get_mirror_init_plane_from_3points(p1, p2, p3, plane_init_size=1000):
    import open3d as o3d



    plane_parameter = get_planeParam_from_3_points(p1, p2, p3)

    [a, b, c, d] = plane_parameter
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    z = Symbol('z', real=True)
    selected_center = np.mean([p1, p2, p3], axis=0)
    try:
        camera_plane_p1, camera_plane_p2 = solve([a * x + b * y + c * z + d,
                                                  (x - selected_center[0]) * (x - selected_center[0]) + (
                                                          y - selected_center[1]) * (y - selected_center[1]) + (
                                                          z - selected_center[2]) * (
                                                          z - selected_center[2]) - 1000 * 1000,
                                                  y - selected_center[1] - 800], [x, y, z])
        camera_plane_p3, camera_plane_p4 = solve([a * x + b * y + c * z + d,
                                                  (x - selected_center[0]) * (x - selected_center[0]) + (
                                                          y - selected_center[1]) * (y - selected_center[1]) + (
                                                          z - selected_center[2]) * (
                                                          z - selected_center[2]) - 1000 * 1000,
                                                  y - selected_center[1] + 800], [x, y, z])
    except:
        camera_plane_p1, camera_plane_p2 = solve([a * x + b * y + c * z + d,
                                                  (x - selected_center[0]) * (x - selected_center[0]) + (
                                                          y - selected_center[1]) * (y - selected_center[1]) + (
                                                          z - selected_center[2]) * (
                                                          z - selected_center[2]) - 1000 * 1000,
                                                  y - selected_center[1] - 1], [x, y, z])
        camera_plane_p3, camera_plane_p4 = solve([a * x + b * y + c * z + d,
                                                  (x - selected_center[0]) * (x - selected_center[0]) + (
                                                          y - selected_center[1]) * (y - selected_center[1]) + (
                                                          z - selected_center[2]) * (
                                                          z - selected_center[2]) - 1000 * 1000,
                                                  y - selected_center[1] + 1], [x, y, z])

    camera_plane_p1 = [float(complex(i).real) for i in camera_plane_p1]
    camera_plane_p2 = [float(complex(i).real) for i in camera_plane_p2]
    camera_plane_p3 = [float(complex(i).real) for i in camera_plane_p3]
    camera_plane_p4 = [float(complex(i).real) for i in camera_plane_p4]

    camera_plane = o3d.geometry.TriangleMesh()
    camera_plane.vertices = o3d.utility.Vector3dVector(
        np.array([camera_plane_p1, camera_plane_p2, camera_plane_p3, camera_plane_p4]))
    camera_plane.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 1, 0], [3, 1, 0], [3, 2, 1]]))
    camera_plane.paint_uniform_color([0.1, 1, 1])
    return camera_plane




# ---------------------------------------------------------------------------- #
#                     get_mirror_init_plane_from_mirrorbbox                    #
# ---------------------------------------------------------------------------- #
def get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox):
    import open3d as o3d
    import itertools

    mirror_bbox_points = np.array(mirror_bbox.get_box_points()).tolist()

    p1 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p1)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p1))
    p2 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p2)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p2))
    p3 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p3)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p3))
    p4 = mirror_bbox_points[0]

    x_max = max(p1[0], p2[0], p3[0], p4[0])
    x_min = min(p1[0], p2[0], p3[0], p4[0])
    y_max = max(p1[1], p2[1], p3[1], p4[1])
    y_min = min(p1[1], p2[1], p3[1], p4[1])

    # y_gap = (y_max - y_min)*0.1
    # y_max += y_gap
    # y_min -= y_gap

    # plane_p1 = [x_max, y_max, get_z_from_plane(plane_parameter, x_max, y_max)]
    # plane_p2 = [x_max, y_min, get_z_from_plane(plane_parameter, x_max, y_min)]
    # plane_p3 = [x_min, y_min, get_z_from_plane(plane_parameter, x_min, y_max)]
    # plane_p4 = [x_min, y_max, get_z_from_plane(plane_parameter, x_min, y_min)]

    plane_p1 = [p1[0], p1[1], get_z_from_plane(plane_parameter, p1[0], p1[1])]
    plane_p2 = [p2[0], p2[1], get_z_from_plane(plane_parameter, p2[0], p2[1])]
    plane_p3 = [p3[0], p3[1], get_z_from_plane(plane_parameter, p3[0], p3[1])]
    plane_p4 = [p4[0], p4[1], get_z_from_plane(plane_parameter, p4[0], p4[1])]
    
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(np.array([plane_p1, plane_p2, plane_p3, plane_p4]))
    plane.triangles = o3d.utility.Vector3iVector(
        np.array(list(itertools.permutations([0,1,2,3],3))))
    plane.paint_uniform_color([0.5, 1, 1])
    plane = resize_plane(plane, 1.2)
    return plane


# ---------------------------------------------------------------------------- #
#                     resize_plane                                             #
# ---------------------------------------------------------------------------- #
def resize_plane(plane, ratio):
    import open3d as o3d

    p1, p2, p3, p4 = np.unique(np.array(plane.vertices), axis=0)

    plane_parameter = get_planeParam_from_3_points(p1, p2, p3)
    [a, b, c, d] = plane_parameter
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    z = Symbol('z', real=True)

    selected_center = np.mean([p1, p2, p3, p4], axis=0)
    point_center_distance = np.linalg.norm(np.array(p1) - np.array(selected_center)) * ratio

    camera_plane_p1, camera_plane_p2 = solve([a * x + b * y + c * z + d,
                                              (x - selected_center[0]) * (x - selected_center[0]) + (
                                                      y - selected_center[1]) * (y - selected_center[1]) + (
                                                      z - selected_center[2]) * (z - selected_center[
                                                  2]) - point_center_distance * point_center_distance,
                                              (y - selected_center[1]) * (p1[0] - selected_center[0]) - (
                                                      x - selected_center[0]) * (p1[1] - selected_center[1])],
                                             [x, y, z])
    camera_plane_p3, camera_plane_p4 = solve([a * x + b * y + c * z + d,
                                              (x - selected_center[0]) * (x - selected_center[0]) + (
                                                      y - selected_center[1]) * (y - selected_center[1]) + (
                                                      z - selected_center[2]) * (z - selected_center[
                                                  2]) - point_center_distance * point_center_distance,
                                              (y - selected_center[1]) * (p2[0] - selected_center[0]) - (
                                                      x - selected_center[0]) * (p2[1] - selected_center[1])],
                                             [x, y, z])

    camera_plane_p1 = [float(complex(i).real) for i in camera_plane_p1]
    camera_plane_p2 = [float(complex(i).real) for i in camera_plane_p2]
    camera_plane_p3 = [float(complex(i).real) for i in camera_plane_p3]
    camera_plane_p4 = [float(complex(i).real) for i in camera_plane_p4]

    plane.vertices = o3d.utility.Vector3dVector(
        np.array([camera_plane_p1, camera_plane_p2, camera_plane_p3, camera_plane_p4]))
    plane.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 1, 0], [3, 1, 0], [3, 2, 1]]))

    return plane


# ---------------------------------------------------------------------------- #
#                     mask mirror border area in RGB image                     #
# ---------------------------------------------------------------------------- #
def visualize_mask_one_image(color_img_path, mask):
    i = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    # convert to floating point
    img = np.array(i, dtype=np.float)
    img /= 255.0
    # convert to floating point
    mask = cv2.cvtColor(mask.astype(np.uint16), cv2.COLOR_GRAY2RGB)
    mask[mask > 0] = 255
    mask = np.array(mask, dtype=np.float)
    mask /= 255.0
    # set transparency to 25%
    transparency = .25
    mask *= transparency
    # make a green overlay
    green = np.ones(img.shape, dtype=np.float) * (0, 1, 0)
    # green over original image
    masked_img = green * mask + img * (1.0 - mask)

    return masked_img


# ---------------------------------------------------------------------------- #
#                               get_picked_points                              #
# ---------------------------------------------------------------------------- #
def get_picked_points(pcd):
    import open3d as o3d
    while 1:
        coor_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8000, origin=[0, 0, 0])
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        points_index = vis.get_picked_points()
        if len(points_index) < 3:
            print("please pick at least three points !")
        else:
            p1 = np.array(pcd.points)[points_index[0]]
            p2 = np.array(pcd.points)[points_index[1]]
            p3 = np.array(pcd.points)[points_index[2]]
            return [p1, p2, p3]


# ---------------------------------------------------------------------------- #
#                         get_planeParam_from_3_points                         #
# ---------------------------------------------------------------------------- #
def get_planeParam_from_3_points(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    # d = -np.dot(cp, p3)
    d = -a * p1[0] - b * p1[1] - c * p1[2]
    return [a, b, c, d]


# ---------------------------------------------------------------------------- #
#            refine_pcd based on mirror border and mirror parameter            #
# ---------------------------------------------------------------------------- #
def refine_pcd_by_mirror_border(mirror_mask, mirror_border_mask, depth_img_path, color_img_path, f):
    import open3d as o3d
    if mirror_mask is not None and len(mirror_mask.shape) > 2:
        mirror_mask = cv2.cvtColor(mirror_mask, cv2.COLOR_BGR2GRAY)
    if mirror_border_mask is not None and len(mirror_border_mask.shape) > 2:
        mirror_border_mask = cv2.cvtColor(mirror_border_mask, cv2.COLOR_BGR2GRAY)

    depth = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img / 255

    h, w = depth.shape
    correct_colors = []
    correct_xyz = []
    border_xyz = []

    for y in range(h):
        for x in range(w):
            if mirror_mask[y][x] == 0:
                correct_colors.append(color_img[y][x])
                correct_xyz.append([(x - w / 2) * (depth[y][x] / f), (y - h / 2) * (depth[y][x] / f), depth[y][x]])
                if mirror_border_mask[y][x] > 0 and depth[y][x] > 10:
                    border_xyz.append([(x - w / 2) * (depth[y][x] / f), (y - h / 2) * (depth[y][x] / f), depth[y][x]])
            else:
                correct_colors.append([1, 0, 0])
                correct_xyz.append([(x - w / 2) * (depth[y][x] / f), (y - h / 2) * (depth[y][x] / f), depth[y][x]])

    try:
        a, b, c, d = get_mirror_parameter_from_xyzs_by_ransac(border_xyz)
    except:
        a, b, c, d = [0, 0, 1, 1000]

    for y in range(h):
        for x in range(w):
            if mirror_mask[y][x] > 0:
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u

                correct_xyz.append(list(I))
                correct_colors.append([0, 0.9, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(correct_xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(correct_colors, axis=0))

    return pcd, [a, b, c, d]


# ---------------------------------------------------------------------------- #
#                          get points in masked area from pcd                  #
# ---------------------------------------------------------------------------- #
def get_points_in_mask(f, depth_img_path, mirror_mask=None, points_num=None):
    import open3d as o3d
    d = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

    h, w = d.shape
    x_cam = []
    y_cam = []
    z_cam = []
    xyz = []

    for y in range(h):
        for x in range(w):
            if mirror_mask is not None and mirror_mask[y][x]:
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])
                if points_num is not None and len(xyz) >= points_num:
                    return xyz

    return xyz


# ---------------------------------------------------------------------------- #
#                           get_pcd_from_rgb_depthMap                          #
# ---------------------------------------------------------------------------- #
def get_pcd_from_rgb_depthMap(f, d, color_img_path, mirror_mask=None, color=None):
    import open3d as o3d
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img / 255

    h, w = d.shape
    x_cam = []
    y_cam = []
    z_cam = []
    colors = []
    xyz = []

    for y in range(h):
        for x in range(w):
            if mirror_mask is not None and mirror_mask[y][x]:
                if color is not None:
                    colors.append(color)
                else:
                    colors.append([0, 0, 1])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])
            else:
                if color is not None:
                    colors.append(color)
                else:
                    colors.append(color_img[y][x])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors, axis=0))
    return pcd

def get_3d_from_2d_array(points_2d, depth_img_path, f, plane_parameter):
    points_3d = []
    depth = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    img_h, img_w = depth.shape
    for one_2d_point in points_2d:
        one_2d_point = one_2d_point[0]
        x = int(one_2d_point[0])
        y = int(one_2d_point[1])

        if x > 0 and x < img_w and y > 0 and y < img_h:
            z = depth[x][y]
            x = (int(one_2d_point[0]) - img_w / 2) * (depth[int(one_2d_point[1])][int(one_2d_point[0])] / f)
            y = (int(one_2d_point[1]) - img_h / 2) * (depth[int(one_2d_point[1])][int(one_2d_point[0])] / f)
            pz = get_z_from_plane(plane_parameter, x, y)
            points_3d.append([x, y, pz])

    return points_3d


def clamp_points_2_plane(ori_points_3d, plane_parameter):
    points_3d = []
    for one_3d_point in ori_points_3d:
        x = one_3d_point[0]
        y = one_3d_point[1]
        pz = get_z_from_plane(plane_parameter, x, y)
        points_3d.append([x, y, pz])

    return points_3d




# ---------------------------------------------------------------------------- #
#                               get_pcd_from_rgbd_depthPath                    #
# ---------------------------------------------------------------------------- #
def get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path, mirror_mask=None, color=None, w=None, h=None):
    import open3d as o3d
    d = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
    color_img = color_img / 255

    if w is not None and h is not None:
        color_img = Image.open(color_img_path).resize((w, h), Image.NEAREST)
        color_img = np.asarray(color_img, dtype=np.float32) / 255.0
        d = np.asarray(cv2.resize(d, dsize=(w, h), interpolation=cv2.INTER_NEAREST), dtype=np.float32)
    else:
        h, w = d.shape

    x_cam = []
    y_cam = []
    z_cam = []
    colors = []
    xyz = []

    for y in range(h):
        for x in range(w):
            if mirror_mask is not None and mirror_mask[y][x]:
                if color is not None:
                    colors.append(color)
                else:
                    colors.append([0, 0, 1])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])
            else:
                colors.append(color_img[y][x])
                xyz.append([(x - w / 2) * (d[y][x] / f), (y - h / 2) * (d[y][x] / f), d[y][x]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors, axis=0))
    return pcd


# ---------------------------------------------------------------------------- #
#                              class RefineDepth                               #
# ---------------------------------------------------------------------------- #
class RefineDepth:
    def __init__(self, focal_len=540, border_width=50, width=640, height=480):
        self.focal_len = focal_len
        self.border_width = border_width
        self.width = width
        self.height = height

    def refine_depth_by_mirror_area(self, instance_mask, plane_normal, np_depth):
        instance_mask = cv2.resize(instance_mask.astype("uint8"), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        instance_mask = instance_mask.astype(bool)
        np_depth = cv2.resize(np_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)

        # plane : ax + by + cd + d = 0
        self.height, self.width = instance_mask.shape
        a, b, c = plane_normal
        offset = (np_depth * instance_mask).sum() / instance_mask.sum()
        py = np.where(instance_mask)[0].mean()
        px = np.where(instance_mask)[1].mean()
        x0 = (px - self.width / 2) * (offset / self.focal_len)
        y0 = (py - self.height / 2) * (offset / self.focal_len)
        d = -(a * x0 + b * y0 + c * offset)
        for y in range(self.height):
            for x in range(self.width):
                if instance_mask[y][x]:
                    n = np.array([a, b, c])
                    # plane function : ax + by + cz + d = 0 ---> x = 0 , y = 0 , c = -d/c
                    V0 = np.array([0, 0, -d / c])
                    P0 = np.array([0, 0, 0])
                    P1 = np.array([(x - self.width / 2), (y - self.height / 2), self.focal_len])

                    j = P0 - V0
                    u = P1 - P0
                    N = -np.dot(n, j)
                    D = np.dot(n, u)
                    sI = N / D
                    I = P0 + sI * u

                    np_depth[y, x] = I[2]
        return np_depth

    def refine_depth_by_mirror_border(self, instance_mask, plane_normal, np_depth):
        # plane : ax + by + cd + d = 0
        instance_mask = cv2.resize(instance_mask.astype("uint8"), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        instance_mask = instance_mask.astype(bool)
        np_depth = cv2.resize(np_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        self.height, self.width = instance_mask.shape
        a, b, c = plane_normal
        new_mask = cv2.dilate(np.array(instance_mask).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.border_width, self.border_width)))
        mirror_border_mask = new_mask - instance_mask
        offset = (np_depth * mirror_border_mask).sum() / mirror_border_mask.sum()
        py = np.where(instance_mask)[0].mean()
        px = np.where(instance_mask)[1].mean()
        x0 = (px - self.width / 2) * (offset / self.focal_len)
        y0 = (py - self.height / 2) * (offset / self.focal_len)
        d = -(a * x0 + b * y0 + c * offset)
        for y in range(self.height):
            for x in range(self.width):
                if instance_mask[y][x]:
                    n = np.array([a, b, c])
                    # plane function : ax + by + cz + d = 0 ---> x = 0 , y = 0 , c = -d/c
                    V0 = np.array([0, 0, -d / c])
                    P0 = np.array([0, 0, 0])
                    P1 = np.array([(x - self.width / 2), (y - self.height / 2), self.focal_len])

                    j = P0 - V0
                    u = P1 - P0
                    N = -np.dot(n, j)
                    D = np.dot(n, u)
                    sI = N / D
                    I = P0 + sI * u

                    np_depth[y, x] = I[2]
        return np_depth


# ---------------------------------------------------------------------------- #
#                                get_normal_vis                                #
# ---------------------------------------------------------------------------- #
def get_normal_vis(cfg, colors, gt_annotations, pred_instances, img_save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    color_list = ["orange", "purple", "pink", "yellow"]
    fig = plt.figure()

    # ----------- XYZ camera coodinate -----------
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-2000, 2000)
    ax.set_ylim3d(-2000, 2000)
    ax.set_zlim3d(-2000, 2000)
    ax.quiver(0, 0, 0, 0, 0, 1, length=2000, color='b', arrow_length_ratio=0.2)  # Z
    ax.text(0, 0, 2000, "z", color='b')
    ax.quiver(0, 0, 0, 0, 1, 0, length=2000, color='g', arrow_length_ratio=0.2)  # Y
    ax.text(0, 2000, 0, "y", color='g')
    ax.quiver(0, 0, 0, 1, 0, 0, length=2000, color='r', arrow_length_ratio=0.2)  # X
    ax.text(2000, 0, 0, "x", color='r')

    anchor_normals = np.load(cfg.ANCHOR_NORMAL_NYP)
    # ----------- GT anchor normal -----------
    for anchor_id, one_normal in enumerate(anchor_normals):
        # anchor normal
        ax.quiver(0, 0, 0, one_normal[0], one_normal[1], one_normal[2], length=2000, color=color_list[anchor_id],
                  arrow_length_ratio=0.2)
        ax.text(one_normal[0] * 2000, one_normal[1] * 2000, one_normal[2] * 2000, str(anchor_id),
                color=color_list[anchor_id], fontsize=8)

    # ----------- pred mirror_normal_camera -----------
    for idx in range(len(pred_instances.pred_anchor_classes)):
        if pred_instances.pred_anchor_classes[idx] == cfg.ANCHOR_NORMAL_CLASS_NUM:  # background
            one_pred_mirror_normal = pred_instances.pred_residuals[idx].detach().cpu().numpy()
            ax.text(one_pred_mirror_normal[0] * 2000, one_pred_mirror_normal[1] * 2000,
                    one_pred_mirror_normal[2] * 2000, "no_pred", color=colors[idx], fontsize=15)
            continue
        one_pred_mirror_normal = anchor_normals[pred_instances.pred_anchor_classes[idx]] + \
                                 pred_instances.pred_residuals[idx].detach().cpu().numpy()
        ax.quiver(0, 0, 0, one_pred_mirror_normal[0], one_pred_mirror_normal[1], one_pred_mirror_normal[2], length=2000,
                  color=colors[idx], arrow_length_ratio=0.2)

        # ----------- GT mirror_normal_camera -----------
    for one_annotation in gt_annotations:
        one_gt_mirror_normal = anchor_normals[one_annotation["anchor_normal_class"]] + one_annotation[
            "anchor_normal_residual"]
        ax.quiver(0, 0, 0, one_gt_mirror_normal[0], one_gt_mirror_normal[1], one_gt_mirror_normal[2], length=1000,
                  color=[0, 0.9, 0], arrow_length_ratio=0.5)

    degree = (angle_between(one_gt_mirror_normal, one_pred_mirror_normal) / np.pi) * 180
    plt.title("{:.2f}".format(degree), fontsize=40)
    ax.view_init(-60, -60)

    plt.savefig(img_save_path)
    print("get_normal_vis saved to : ", img_save_path)


# ---------------------------------------------------------------------------- #
#                                 draw_gt_bbox                                 #
# ---------------------------------------------------------------------------- #
def draw_gt_bbox(annotations, img, pred_anchor_classes):
    import numpy as np
    img = np.array(img)
    predict_correct = False
    for one_annotation in annotations:

        gt_anchor_normal_class = one_annotation["anchor_normal_class"]
        gt_bbox = one_annotation["bbox"]
        if len(annotations) == 1 and len(pred_anchor_classes) == 1 and gt_anchor_normal_class in pred_anchor_classes:
            predict_correct = True
        cv2.rectangle(img, (int(gt_bbox[0] / 2), int(gt_bbox[1] / 2)),
                      (int(gt_bbox[0] / 2) + int(gt_bbox[2] / 2), int(gt_bbox[1] / 2) + int(gt_bbox[3] / 2)),
                      (0, 255, 0), 1)
        cv2.putText(img, str(gt_anchor_normal_class), (int(gt_bbox[0] / 2), int(gt_bbox[1] / 2 + 20)), 1, 2,
                    (0, 255, 0))
    return img, predict_correct


# ---------------------------------------------------------------------------- #
#                              plane_pcd_interact                              #
# ---------------------------------------------------------------------------- #
# ----------------------- requirement : open3d 0.10.0 + ---------------------- #
def get_parameter_from_plane_adjustment(pcd, camera_plane, adjustment_init_step_size):
    import open3d as o3d
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5000, origin=[0, 0, 0])
    coor.translate(np.array(camera_plane.vertices).mean(0), relative=False)

    init_rotation_angle = 0.5
    up = False
    down = False
    left = False
    right = False
    forward = False
    backward = False

    turn_up = False
    turn_down = False
    turn_right = False
    turn_left = False
    shrink = False
    expand = False

    vis = o3d.visualization.VisualizerWithKeyCallback()

    # ------------------------ define function for one key ----------------------- #
    def mirror_up(vis, action, mods):
        nonlocal up
        if action == 1:  # key down
            up = True
        elif action == 0:  # key up
            up = False
        elif action == 2:  # key repeat
            up = True
        return True

    def mirror_down(vis, action, mods):
        nonlocal down
        if action == 1:  # key down
            down = True
        elif action == 0:  # key up
            down = False
        elif action == 2:  # key repeat
            down = True
        return True

    def mirror_left(vis, action, mods):
        nonlocal left
        if action == 1:  # key down
            left = True
        elif action == 0:  # key up
            left = False
        elif action == 2:  # key repeat
            left = True
        return True

    def mirror_right(vis, action, mods):
        nonlocal right
        if action == 1:  # key down
            right = True
        elif action == 0:  # key up
            right = False
        elif action == 2:  # key repeat
            right = True
        return True

    def mirror_forward(vis, action, mods):
        nonlocal forward
        if action == 1:  # key down
            forward = True
        elif action == 0:  # key up
            forward = False
        elif action == 2:  # key repeat
            forward = True
        return True

    def mirror_backward(vis, action, mods):
        nonlocal backward
        if action == 1:  # key down
            backward = True
        elif action == 0:  # key up
            backward = False
        elif action == 2:  # key repeat
            backward = True
        return True

    def mirror_turn_up(vis, action, mods):
        nonlocal turn_up
        if action == 1:  # key down
            turn_up = True
        elif action == 0:  # key up
            turn_up = False
        elif action == 2:  # key repeat
            turn_up = True
        return True

    def mirror_turn_down(vis, action, mods):
        nonlocal turn_down
        if action == 1:  # key down
            turn_down = True
        elif action == 0:  # key up
            turn_down = False
        elif action == 2:  # key repeat
            turn_down = True
        return True

    def mirror_turn_left(vis, action, mods):
        nonlocal turn_left
        if action == 1:  # key down
            turn_left = True
        elif action == 0:  # key up
            turn_left = False
        elif action == 2:  # key repeat
            turn_left = True
        return True

    def mirror_turn_right(vis, action, mods):
        nonlocal turn_right
        if action == 1:  # key down
            turn_right = True
        elif action == 0:  # key up
            turn_right = False
        elif action == 2:  # key repeat
            turn_right = True
        return True

    def mirror_expand(vis, action, mods):
        nonlocal expand
        if action == 1:  # key down
            expand = True
        elif action == 0:  # key up
            expand = False
        elif action == 2:  # key repeat
            expand = True
        return True

    def mirror_shrink(vis, action, mods):
        nonlocal shrink
        if action == 1:  # key down
            shrink = True
        elif action == 0:  # key up
            shrink = False
        elif action == 2:  # key repeat
            shrink = True
        return True

    # ------------------------- define callback action ------------------------ #
    def animation_callback(vis):
        if up:
            camera_plane.translate((0, adjustment_init_step_size, 0))
            vis.update_geometry(camera_plane)
        if down:
            camera_plane.translate((0, -adjustment_init_step_size, 0))
            vis.update_geometry(camera_plane)
        if left:
            camera_plane.translate((adjustment_init_step_size, 0, 0))
            vis.update_geometry(camera_plane)
        if right:
            camera_plane.translate((-adjustment_init_step_size, 0, 0))
            vis.update_geometry(camera_plane)
        if forward:
            camera_plane.translate((0, 0, adjustment_init_step_size))
            vis.update_geometry(camera_plane)
        if backward:
            camera_plane.translate((0, 0, -adjustment_init_step_size))
            vis.update_geometry(camera_plane)
        if turn_up:
            camera_plane.rotate(get_3_3_rotation_matrix(init_rotation_angle, 0, 0),
                                np.array(camera_plane.vertices).mean(0))
            vis.update_geometry(camera_plane)
        if turn_down:
            camera_plane.rotate(get_3_3_rotation_matrix(-init_rotation_angle, 0, 0),
                                np.array(camera_plane.vertices).mean(0))
            vis.update_geometry(camera_plane)
        if turn_left:
            camera_plane.rotate(get_3_3_rotation_matrix(0, -init_rotation_angle, 0),
                                np.array(camera_plane.vertices).mean(0))
            vis.update_geometry(camera_plane)
        if turn_right:
            camera_plane.rotate(get_3_3_rotation_matrix(0, init_rotation_angle, 0),
                                np.array(camera_plane.vertices).mean(0))
            vis.update_geometry(camera_plane)
        if expand:
            resize_plane(plane=camera_plane, ratio=2)
        if shrink:
            resize_plane(plane=camera_plane, ratio=0.5)

    # ------------------------- link action with key ------------------------ #
    # a 65 w 87 s 83 d 68 e 69 r 82     j 74 i 73 k 75 l 76 o 79 p 80 z 90 x 88 n 78 m 77
    # translation
    vis.register_key_action_callback(65, mirror_left)  # a
    vis.register_key_action_callback(87, mirror_up)  # w
    vis.register_key_action_callback(83, mirror_down)  # s
    vis.register_key_action_callback(68, mirror_right)  # d
    vis.register_key_action_callback(69, mirror_backward)  # e move close
    vis.register_key_action_callback(82, mirror_forward)  # r move far

    # # rotation
    vis.register_key_action_callback(73, mirror_expand)  # i  mirror_expand
    vis.register_key_action_callback(75, mirror_shrink)  # k  mirror_shrink
    vis.register_key_action_callback(74, mirror_turn_left)  # j green_y 
    vis.register_key_action_callback(76, mirror_turn_right)  # l green_y
    vis.register_key_action_callback(79, mirror_turn_up)  # o blue_z
    vis.register_key_action_callback(80, mirror_turn_down)  # p blue_z

    option_list = ToolOption()
    option_list.add_option("a", "plane move left")
    option_list.add_option("w", "plane move up")
    option_list.add_option("s", "plane move down")
    option_list.add_option("d", "plane move right")
    option_list.add_option("e", "plane move closer")
    option_list.add_option("r", "plane move further")
    option_list.add_option("i", "make the plane larger")
    option_list.add_option("k", "make the plane smaller")
    option_list.add_option("j", "rotate left")
    option_list.add_option("l", "rotate right")
    option_list.add_option("o", "rotate upwards")
    option_list.add_option("p", "rotate downwards")
    option_list.print_option()

    coor_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8000, origin=[0, 0, 0])
    vis.register_animation_callback(animation_callback)
    vis.create_window()
    vis.add_geometry(camera_plane)
    vis.add_geometry(pcd)
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_constant_z_far(100000)
    vis.get_view_control().set_constant_z_near(0)
    vis.get_view_control().set_up([0, -1, 0])
    vis.run()

    p1, p2, p3 = np.array(camera_plane.vertices)[:3]

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    # d = -np.dot(cp, p3)
    d = -a * p1[0] - b * p1[1] - c * p1[2]
    return [a, b, c, d]


# ---------------------------------------------------------------------------- #
#                    refine_depth_with_plane_parameter_mask                    #
# ---------------------------------------------------------------------------- #
def refine_depth_with_plane_parameter_mask(plane_parameter, mirror_mask, depth_to_refine, f):
    if mirror_mask is not None and len(mirror_mask.shape) > 2:
        mirror_mask = cv2.cvtColor(mirror_mask, cv2.COLOR_BGR2GRAY)
    h, w = depth_to_refine.shape
    correct_colors = []
    correct_xyz = []
    border_xyz = []
    a, b, c, d = plane_parameter
    for y in range(h):
        for x in range(w):
            if mirror_mask[y][x] > 0:
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u

                depth_to_refine[y, x] = I[2]
    return depth_to_refine


# ---------------------------------------------------------------------------- #
#                          clamp data based on 3d bbox                         #
# ---------------------------------------------------------------------------- #
def clamp_pcd_by_bbox(mirror_bbox, depth_img_path, f, mirror_border_mask, plane_parameter, expand_range=100,
                      clamp_dis=100):
    mirror_bbox_points = np.array(mirror_bbox.get_box_points()).tolist()

    p1 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p1)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p1))
    p2 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p2)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p2))
    p3 = mirror_bbox_points[0]
    mirror_bbox_points.remove(p3)
    mirror_bbox_points.remove(get_paired_point(mirror_bbox_points, p3))
    p4 = mirror_bbox_points[0]
    mirror_recrangle = [p1, p2, p3, p4]

    if mirror_border_mask is not None and len(mirror_border_mask.shape) > 2:
        mirror_border_mask = cv2.cvtColor(mirror_border_mask, cv2.COLOR_BGR2GRAY)

    depth_to_refine = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
    h, w = depth_to_refine.shape
    a, b, c, d = plane_parameter
    for y in range(h):
        for x in range(w):
            if mirror_border_mask[y][x] > 0:
                ori_point = [(x - w / 2) * (depth_to_refine[y][x] / f), (y - h / 2) * (depth_to_refine[y][x] / f),
                             depth_to_refine[y][x]]
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u
                expand_point_on_plane = I
                if point_2_regBorder_in_3d(expand_point_on_plane, mirror_recrangle) <= expand_range:
                    if np.linalg.norm(np.array(expand_point_on_plane) - np.array(ori_point[0])) >= clamp_dis:
                        depth_to_refine[y, x] = I[2]

    return depth_to_refine


# ---------------------------------------------------------------------------- #
#                                 Option class                                 #
# ---------------------------------------------------------------------------- #
class ToolOption:
    """
    The Option class currently does the following:
    1. add_option
    2. print_option
    3. check input_option correctness
    """

    def __init__(self):
        self.option_fun = dict()

    def add_option(self, option_key, option_description):
        self.option_fun[option_key] = option_description

    def print_option(self):
        print("OPTION : ")
        for index, item in enumerate(self.option_fun.items()):
            print("({}) {:8} : {}".format(index + 1, item[0], item[1]))

    def is_input_key_valid(self, input_option):
        key = input_option.split()[0]
        is_valid = False
        for item in self.option_fun.items():
            if key == item[0].split()[0]:
                is_valid = True

        return is_valid


# ---------------------------------------------------------------------------- #
#                         get 3D points' 2D coordinate                         #
# ---------------------------------------------------------------------------- #
def get_2D_coor_from_3D(points_3d, f, w, h):
    points_2d = []
    for point in points_3d:
        points_2d.append([(point[0] * f) / point[2] + w / 2, (point[1] * f) / point[2] + h / 2])
    return points_2d


# ---------------------------------------------------------------------------- #
#                               get_triangle_mask                              #
# ---------------------------------------------------------------------------- #
def get_triangle_mask(points_2d, width, height):
    """
    Args:
        points_2d : 3 points (under 2D coordinate)
    Output:
        triangle_mask : binary mask
    """

    polygon = [tuple(points_2d[0]), tuple(points_2d[1]), tuple(points_2d[2])]
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask


# ---------------------------------------------------------------------------- #
#                  clamp the points in mask and over clamp_dis                 #
# ---------------------------------------------------------------------------- #
def clamp_pcd_by_mask(depth_to_refine, f, clamp_mask, plane_parameter, clamp_dis=100):
    if clamp_mask is not None and len(clamp_mask.shape) > 2:
        clamp_mask = cv2.cvtColor(clamp_mask, cv2.COLOR_BGR2GRAY)

    h, w = depth_to_refine.shape
    a, b, c, d = plane_parameter
    for y in range(h):
        for x in range(w):
            if clamp_mask[y][x] > 0:
                ori_point = [(x - w / 2) * (depth_to_refine[y][x] / f), (y - h / 2) * (depth_to_refine[y][x] / f),
                             depth_to_refine[y][x]]
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u
                expand_point_on_plane = I
                if np.linalg.norm(np.array(expand_point_on_plane) - np.array(ori_point[0])) >= clamp_dis:
                    depth_to_refine[y, x] = I[2]

    return depth_to_refine


# ---------------------------------------------------------------------------- #
#                    get mirror pcd based on plane parameter                   #
# ---------------------------------------------------------------------------- #
def get_mirrorPoint_based_on_plane_parameter(f, plane_parameter=[1, 1, 1, 1], mirror_mask=None, color=None,
                                             color_img_path=""):
    import open3d as o3d
    h, w = cv2.imread(color_img_path, cv2.IMREAD_ANYDEPTH).shape
    a, b, c, d = plane_parameter

    xyz = []
    colors = []

    for y in range(h):
        for x in range(w):
            if mirror_mask[y][x] > 0:
                n = np.array([a, b, c])
                V0 = np.array([0, 0, -d / c])
                P0 = np.array([0, 0, 0])
                P1 = np.array([(x - w / 2), (y - h / 2), f])

                j = P0 - V0
                u = P1 - P0
                N = -np.dot(n, j)
                D = np.dot(n, u)
                sI = N / D
                I = P0 + sI * u

                xyz.append(list(I))
                if color is None:
                    colors.append([0, 0.9, 0])
                else:
                    colors.append(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(xyz, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors, axis=0))
    return pcd
