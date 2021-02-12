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

def get_z_from_plane(plane_parameter, x, y):
    [a, b, c, d] = plane_parameter
    z = (-d - a * x - b * y) / c
    return z

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return ((math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))/np.pi)*180


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    
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

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    # print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic, "used time : ", time.time() - start_time)
    return best_model, best_ic


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_3_3_rotation_matrix(rx, ry, rz):
    rx = (rx/180)*np.pi
    ry = (ry/180)*np.pi
    rz = (rz/180)*np.pi
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
    
    FP = ((pred_mask>0)==(GT_mask==0)).sum()
    FN = ((pred_mask==0)==(GT_mask>0)).sum()
    h, w = GT_mask.shape
    MAE = (FN + FP) / (w * h)

    return MAE


def get_f_measure(pred_mask, GT_mask):
    
    TP = ((pred_mask>0)==(GT_mask>0)).sum()
    FP = ((pred_mask>0)==(GT_mask==0)).sum()
    FN = ((pred_mask==0)==(GT_mask>0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    beta = 0.3
    f_measure = ((1+beta**2)*precision*recall )/((beta**2) * precision + recall)

    return f_measure


def get_IOU(pred_mask, GT_mask):
    intersct = np.logical_and(pred_mask,GT_mask)
    union =  np.logical_or(pred_mask, GT_mask)
    IoU = np.sum(intersct) / np.sum(union)
    return IoU


# ---------------------------------------------------------------------------- #
#                     get point to rectangle distance in 3d                    #
# ---------------------------------------------------------------------------- #
def point_2_regBorder_in_3d(point, rectangle):
    p1, p2, p3, p4 = sorted_rect(rectangle)
    return min(point_2_line_seg_in_3d(point, p1, p2), point_2_line_seg_in_3d(point, p2, p3), point_2_line_seg_in_3d(point, p3, p4), point_2_line_seg_in_3d(point, p4, p1))


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
#             returns vec in clockwise order, starting with topleft            #
# ---------------------------------------------------------------------------- #
def sorted_rect(vec):
    normd = vec - np.average(vec,axis=0) # vertices relative to centroid
    tl_idx = np.argmax(np.dot(normd,np.array([-1,-1]))) #index of top left vertex
    clockwise = np.cross(vec[(tl_idx+1) % 4] - vec[tl_idx],
                         vec[tl_idx] - vec[tl_idx-1]) > 0
    return np.roll(vec,-tl_idx,axis=0) if clockwise else np.roll(vec,-1-tl_idx,axis=0)[::-1]