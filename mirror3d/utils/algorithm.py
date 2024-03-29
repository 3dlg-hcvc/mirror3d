import random
import numpy as np
import math


def get_angle_to_Azimuth(vector):
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


def dot_product(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dot_product(v, v))


def angle(v1, v2):
    return ((math.acos(dot_product(v1, v2) / (length(v1) * length(v2)))) / np.pi) * 180


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    # start_time = time.time()
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
    k_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k_mat + k_mat.dot(k_mat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_MAE(pred_mask, gt_mask):
    fp = ((pred_mask > 0) == (gt_mask == 0)).sum()
    fn = ((pred_mask == 0) == (gt_mask > 0)).sum()
    h, w = gt_mask.shape
    mae = (fn + fp) / (w * h)
    return mae


def get_f_measure(pred_mask, gt_mask):
    tp = ((pred_mask > 0) == (gt_mask > 0)).sum()
    fp = ((pred_mask > 0) == (gt_mask == 0)).sum()
    fn = ((pred_mask == 0) == (gt_mask > 0)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta = 0.3
    f_measure = ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall)
    return f_measure


def get_IOU(pred_mask, gt_mask):
    intersect = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersect) / np.sum(union)
    return iou


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
#             get the closest point to target_point from points_list            #
# ---------------------------------------------------------------------------- #
def get_paired_point(points_list, target_point):
    dis = np.linalg.norm(np.array(target_point) - np.array(points_list[0]))
    closest_point = points_list[0]
    for item in points_list:
        current_dis = np.linalg.norm(np.array(target_point) - np.array(item))
        if current_dis <= dis:
            dis = current_dis
            closest_point = item
    return closest_point


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
