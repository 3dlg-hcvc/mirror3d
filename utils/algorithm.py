import sys
import os
import time
import json
import random
import time
import numpy as np
import cv2
import pdb
import math


def get_angle_to_azimuth(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    hor_angle = np.arctan(x / z)
    hor_degree = (hor_angle / np.pi) * 180
    x_z = math.sqrt(x ** 2 + z ** 2)
    ver_angle = np.arctan(y / x_z)
    ver_degree = (ver_angle / np.pi) * 180
    return hor_degree, ver_degree


def get_extrinsic(rx, ry, rz, translate):
    """
    Args:
        rx, ry, rz are measured in degree
    Output:
        4*4 extrinsic matrix
    """
    translate = np.array(translate)
    t = translate.reshape(3, 1)

    rx = (rx / 180) * np.pi
    ry = (ry / 180) * np.pi
    rz = (rz / 180) * np.pi
    Rx = np.array((1, 0, 0,
                   0, np.cos(rx), -np.sin(rx),
                   0, np.sin(rx), np.cos(rx))).reshape(3, 3)
    Ry = np.array((np.cos(ry), 0, np.sin(ry),
                   0, 1, 0,
                   -np.sin(ry), 0, np.cos(ry))).reshape(3, 3)
    Rz = np.array((np.cos(rz), -np.sin(rz), 0,
                   np.sin(rz), np.cos(rz), 0,
                   0, 0, 1)).reshape(3, 3)
    R = np.dot(Ry, Rx)
    R = np.dot(Rz, R)

    R = np.concatenate([R, t], axis=1)
    last_row = np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, 4)
    R = np.concatenate([R, last_row], axis=0)
    return R


def get_z_from_plane(plane_parameter, x, y):
    [a, b, c, d] = plane_parameter
    z = (-d - a * x - b * y) / c
    return z


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return ((math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))) / np.pi) * 180


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    start_time = time.time()
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    # print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic, "used time : ", time.time() -
    # start_time)
    return best_model, best_ic


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_3_3_rotation_matrix(rx, ry, rz):
    """
    Args:
        rx, ry, rz are measured in degree
    """
    rx = (rx / 180) * np.pi
    ry = (ry / 180) * np.pi
    rz = (rz / 180) * np.pi
    Rx = np.array((1, 0, 0,
                   0, np.cos(rx), -np.sin(rx),
                   0, np.sin(rx), np.cos(rx))).reshape(3, 3)
    Ry = np.array((np.cos(ry), 0, np.sin(ry),
                   0, 1, 0,
                   -np.sin(ry), 0, np.cos(ry))).reshape(3, 3)
    Rz = np.array((np.cos(rz), -np.sin(rz), 0,
                   np.sin(rz), np.cos(rz), 0,
                   0, 0, 1)).reshape(3, 3)
    R = np.dot(Ry, Rx)
    R = np.dot(Rz, R)
    return R


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_MAE(pred_mask, GT_mask):
    FP = ((pred_mask > 0) == (GT_mask == 0)).sum()
    FN = ((pred_mask == 0) == (GT_mask > 0)).sum()
    h, w = GT_mask.shape
    MAE = (FN + FP) / (w * h)

    return MAE


def get_f_measure(pred_mask, GT_mask):
    TP = ((pred_mask > 0) == (GT_mask > 0)).sum()
    FP = ((pred_mask > 0) == (GT_mask == 0)).sum()
    FN = ((pred_mask == 0) == (GT_mask > 0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    beta = 0.3
    f_measure = ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall)

    return f_measure


def get_IOU(pred_mask, GT_mask):
    intersct = np.logical_and(pred_mask, GT_mask)
    union = np.logical_or(pred_mask, GT_mask)
    IoU = np.sum(intersct) / np.sum(union)
    return IoU


# ---------------------------------------------------------------------------- #
#                     get point to rectangle distance in 3d                    #
# ---------------------------------------------------------------------------- #
def point_2_regBorder_in_3d(point, rectangle):
    point = np.array(point)
    p1, p2, p3, p4 = sorted_rect(rectangle)
    return min(point_2_line_seg_in_3d(point, p1, p2), point_2_line_seg_in_3d(point, p2, p3),
               point_2_line_seg_in_3d(point, p3, p4), point_2_line_seg_in_3d(point, p4, p1))


# ---------------------------------------------------------------------------- #
#                       get point to lines distance in 3D                      #
# ---------------------------------------------------------------------------- #
def point_2_line_seg_in_3d(point, line_p1, line_p2):
    # normalized tangent vector
    d = np.divide(line_p2 - line_p1, np.linalg.norm(line_p2 - line_p1))

    # signed parallel distance components
    s = np.dot(line_p1 - point, d)
    t = np.dot(point - line_p2, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(point - line_p1, d)
    return np.hypot(h, np.linalg.norm(c))


# ---------------------------------------------------------------------------- #
#             get the cloest point to target_point from points_list            #
# ---------------------------------------------------------------------------- #
def get_paired_point(points_list, target_point):
    dis = np.linalg.norm(np.array(target_point) - np.array(points_list[0]))
    cloest_point = points_list[0]
    for item in points_list:
        current_dis = np.linalg.norm(np.array(target_point) - np.array(item))
        if current_dis <= dis:
            dis = current_dis
            cloest_point = item
    return cloest_point


# ---------------------------------------------------------------------------- #
#             returns vec in clockwise order, starting with topleft            #
# ---------------------------------------------------------------------------- #
def sorted_rect(vec):
    point_list = vec.copy()
    p2 = point_list[0]
    p1 = get_paired_point(point_list[1:], p2)
    point_list.remove(p1)
    point_list.remove(p2)
    p3 = get_paired_point(point_list, p2)
    point_list.remove(p3)
    p4 = point_list[0]
    return [np.array(p1), np.array(p2), np.array(p3), np.array(p4)]

    def compute_and_update_mirror3D_metrics_new(self, pred_depth, depth_shift, color_image_path):
        if color_image_path.find("m3d") > 0 and "mesh" not in self.Train_tag:
            self.Train_tag = self.Train_tag.replace("ref", "mesh-ref")
            self.Train_tag = self.Train_tag.replace("raw", "mesh")

        def compute_errors(gt, pred, eval_area, mirror_mask, include_mirror):  # ! gt and pred are in m
            gt = np.array(gt, dtype="f")
            pred = np.array(pred, dtype="f")
            min_depth_eval = 1e-3
            max_depth_eval = 10

            pred[pred < min_depth_eval] = min_depth_eval
            pred[np.isinf(pred)] = max_depth_eval

            gt[np.isinf(gt)] = 0
            gt[np.isnan(gt)] = 0

            valid_mask = gt > min_depth_eval  # np.logical_and(gt > min_depth_eval)#, gt < max_depth_eval
            scale = np.sum(pred[valid_mask] * gt[valid_mask]) / np.sum(pred[valid_mask] ** 2)
            valid_mask = np.logical_and(valid_mask, eval_area)
            if include_mirror:
                valid_mask = np.logical_or(valid_mask, mirror_mask)

            SSIM_obj = SSIM()
            ssim_map = SSIM_obj.forward(torch.tensor(pred * valid_mask.astype(int)).unsqueeze(0).unsqueeze(0),
                                        torch.tensor(gt * valid_mask.astype(int)).unsqueeze(0).unsqueeze(0))
            ssim = ssim_map[valid_mask].mean()

            gt = gt[valid_mask]
            pred = pred[valid_mask]

            if valid_mask.sum() == 0 or sum(gt[valid_mask]) == 0:
                return False

            thresh = np.maximum((gt / pred), (pred / gt))
            d125 = (thresh < 1.25).mean()
            d125_2 = (thresh < 1.25 ** 2).mean()
            d125_3 = (thresh < 1.25 ** 3).mean()
            d105 = (thresh < 1.05).mean()
            d110 = (thresh < 1.10).mean()

            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())

            rel = np.mean((abs(gt - pred)) / gt)

            err = np.log(pred) - np.log(gt)
            silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

            err = np.abs(np.log10(pred) - np.log10(gt))
            log10 = np.mean(err)

            scaled_rms = np.sqrt(((scale * pred - gt) ** 2).mean())
            return rmse, scaled_rms, rel, ssim.item(), d105, d110, d125, d125_2, d125_3

        def get_refD_scores(pred_depth, depth_shift, color_image_path):
            mask_path = rreplace(color_image_path, "raw", "instance_mask")
            if not os.path.exists(mask_path):
                return

            if color_image_path.find("m3d") > 0:
                if os.path.exists(rreplace(color_image_path.replace("raw", "mesh_refined_depth"), "i", "d")):
                    refD_gt_depth_path = rreplace(color_image_path.replace("raw", "mesh_refined_depth"), "i", "d")
                elif os.path.exists(rreplace(color_image_path.replace("raw", "hole_refined_depth"), "i", "d")):
                    refD_gt_depth_path = rreplace(color_image_path.replace("raw", "hole_refined_depth"), "i", "d")
                else:
                    return
            else:
                if os.path.exists(color_image_path.replace("raw", "mesh_refined_depth")):
                    refD_gt_depth_path = color_image_path.replace("raw", "mesh_refined_depth")
                elif os.path.exists(color_image_path.replace("raw", "hole_refined_depth")):
                    refD_gt_depth_path = color_image_path.replace("raw", "hole_refined_depth")
                else:
                    return
            depth_shift = np.array(depth_shift)
            refD_gt_depth = cv2.resize(cv2.imread(refD_gt_depth_path, cv2.IMREAD_ANYDEPTH),
                                       (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
            refD_gt_depth = np.array(refD_gt_depth) / depth_shift
            pred_depth = np.array(pred_depth)

            mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0,
                                     cv2.INTER_NEAREST)
            pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
            refD_gt_depth = cv2.resize(refD_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)

            mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask > 0, mirror_mask=mirror_mask > 0,
                                          include_mirror=True)
            non_mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask == False,
                                              mirror_mask=mirror_mask > 0, include_mirror=False)
            all_image_error = compute_errors(refD_gt_depth, pred_depth, True, mirror_mask=mirror_mask > 0,
                                             include_mirror=True)
            if all_image_error == False or mirror_error == False or non_mirror_error == False:
                return
            one_m_nm_all = mirror_error + non_mirror_error + all_image_error
            return one_m_nm_all

        def get_rawD_scores(pred_depth, depth_shift, color_image_path):
            mask_path = rreplace(color_image_path, "raw", "instance_mask")
            if not os.path.exists(mask_path):
                return
            if color_image_path.find("m3d") > 0:
                if os.path.exists(rreplace(color_image_path.replace("raw", "mesh_raw_depth"), "i", "d")):
                    refD_gt_depth_path = rreplace(color_image_path.replace("raw", "mesh_raw_depth"), "i", "d")
                elif os.path.exists(rreplace(color_image_path.replace("raw", "hole_raw_depth"), "i", "d")):
                    refD_gt_depth_path = rreplace(color_image_path.replace("raw", "hole_raw_depth"), "i", "d")
                else:
                    return
            else:
                if os.path.exists(color_image_path.replace("raw", "mesh_raw_depth")):
                    refD_gt_depth_path = color_image_path.replace("raw", "mesh_raw_depth")
                elif os.path.exists(color_image_path.replace("raw", "hole_raw_depth")):
                    refD_gt_depth_path = color_image_path.replace("raw", "hole_raw_depth")
                else:
                    return
            depth_shift = np.array(depth_shift)
            refD_gt_depth = cv2.resize(cv2.imread(refD_gt_depth_path, cv2.IMREAD_ANYDEPTH),
                                       (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
            refD_gt_depth = np.array(refD_gt_depth) / depth_shift
            pred_depth = np.array(pred_depth)

            mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0,
                                     cv2.INTER_NEAREST)
            pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
            refD_gt_depth = cv2.resize(refD_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)

            mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask > 0, mirror_mask=mirror_mask > 0,
                                          include_mirror=True)
            non_mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask == False,
                                              mirror_mask=mirror_mask > 0, include_mirror=False)
            all_image_error = compute_errors(refD_gt_depth, pred_depth, True, mirror_mask=mirror_mask > 0,
                                             include_mirror=True)
            if all_image_error == False or mirror_error == False or non_mirror_error == False:
                return
            one_m_nm_all = mirror_error + non_mirror_error + all_image_error
            return one_m_nm_all

        self.m_nm_all_refD += torch.tensor(get_refD_scores(np.array(pred_depth).copy(), depth_shift, color_image_path))
        self.m_nm_all_rawD += torch.tensor(get_rawD_scores(np.array(pred_depth).copy(), depth_shift, color_image_path))
        self.raw_cnt += 1  # TODO to be update later

        return
