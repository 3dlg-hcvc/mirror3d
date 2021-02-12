def read_mask_from_path(path):
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)

def get_insid_list(mask)
    """
    Get the mirror instances list from given mask
    :param mask: np array that stores the segmentation image
    :return: mirror instances id list
    """
    return np.delete(np.unique(mask), [0])

def align_insid(coarse_mask, precise_mask):
    insid_list = get_insid_list(coarse_mask)
    for insid in insid_list:
        coarse_mask_on_precise = precise_mask[coarse_mask==insid]
        mode_info = stats.mode(coarse_mask_on_precise, axis=None)
        insid_of_precise = mode_info[0]
        if insid_of_precise != insid:
            precise_mask[precise_mask==insid_of_precise] = insid
    return precise_mask

