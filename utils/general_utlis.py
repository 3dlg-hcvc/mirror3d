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
from PIL import Image
import matplotlib.pyplot as plt
from utils.algorithm import *
import shutil


def list_diff(list1, list2):
    """
    Get a list exist in list1 but don't exisit in list2
    """
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)
    return out

def center_crop_image(input_folder="", output_folder = "", new_w=608, new_h=456, ori_w=640, ori_h=480):
    """
    Center crop the image
    Args:
        input_folder : folder path; folder contains the image that needs to be cropped
        output_folder : folder path to saved the cropped result
    """
    os.makedirs(output_folder, exist_ok=True)
    w_border = int((ori_w - new_w)/2)
    h_border = int((ori_h - new_h)/2)
    for one_img in os.listdir(input_folder):
        one_img_path = os.path.join(input_folder, one_img)
        one_img_save_path = os.path.join(output_folder, one_img)
        h, w, _ = cv2.imread(one_img_path).shape
        if h == new_h and w == new_w:
            shutil.copy(one_img_path, one_img_save_path)
            continue
        try:
            if one_img_path.find("depth") > 0:
                ori_img = cv2.imread(one_img_path, cv2.IMREAD_ANYDEPTH) 
                ori_img = ori_img[h_border:h_border+new_h, w_border:w_border+new_w]
            else:
                ori_img = cv2.imread(one_img_path)
                ori_img = ori_img[h_border:h_border+new_h, w_border:w_border+new_w]
        except:
            print(print("error: ", one_img_path))
            continue
        cv2.imwrite(one_img_save_path, ori_img)
    print("corpped image saved to {}".format(output_folder))


def save_html(save_path, content):
    with open(save_path, "w") as outf:
        outf.write(str(content))
    print("html saved to {}".format(save_path))

def save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_index):

    sample_id = "{}_{}_{}".format(instance_index[0], instance_index[1], instance_index[2])
    if os.path.exists(one_plane_para_save_path):
        with open(one_plane_para_save_path, 'r') as j:
            img_info = json.loads(j.read())
    else:
        img_info = dict()
    img_info[str(sample_id)] = dict()
    img_info[str(sample_id)]["plane_parameter"] = list(plane_parameter)
    img_info[str(sample_id)]["mirror_normal"] = list(plane_parameter[:-1])
    angle_degree = angle(np.array(plane_parameter[:-1]), np.array([0,0,1]))
    if angle_degree > 90 and angle_degree <= 180:
        angle_degree = 180 - angle_degree
    img_info[str(sample_id)]["angle"] = angle_degree
    save_json(one_plane_para_save_path,img_info)

def get_all_fileAbsPath_under_folder(folder_path):
    file_path_list = []
    for root, dirs, files in os.walk(os.path.abspath(folder_path)):
        for file in files:
            file_path_list.append(os.path.join(root, file))
    return file_path_list

def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info

def read_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def save_txt(save_path, data):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", save_path, len(data))

def save_json(save_path,data):
    out_json = json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    with open(save_path, "w") as fo:
        fo.write(out_json)
        fo.close()
        print("json file saved to : ",save_path )

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)


def get_compose_image(output_save_path, img_list, mini_img_w=320, mini_img_h=240, mini_image_per_row=9):
    """
    Args:
        img_list : Image Array
        output_save_path : composed image saved path
    """

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    image_col = math.ceil(len(img_list)/mini_image_per_row) 
    to_image = Image.new('RGB', (mini_image_per_row * mini_img_w, image_col * mini_img_h)) 

    for y in range(1, image_col + 1):
        for x in range(1, mini_image_per_row + 1):
            img_index = mini_image_per_row * (y - 1) + x - 1
            from_image = img_list[img_index].resize((mini_img_w, mini_img_h),Image.ANTIALIAS)
            from_image = add_margin(from_image, 20,20,20,20,(255,255,255))
            to_image.paste(from_image, ((x - 1) * mini_img_w, (y - 1) * mini_img_h))
    to_image.save(output_save_path) 
    print("image saved to :", output_save_path)



def save_heatmap_no_border(image, save_path=""):
    """ 
    Save heatmap with no border
    Args:
        image : M * N image 
    """
    plt.figure()
    fig = plt.imshow(image, cmap=plt.get_cmap("magma"))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    figure = plt.gcf()
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=100)
    print("image saved to : {}".format(save_path))



def get_filtered_percantage(dataset="scannet"):
    from tqdm import tqdm
    data_main_path = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/{}".format(dataset)
    test_json = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/{}/with_mirror/precise/network_input_json/test_10_normal_mirror.json".format(dataset)
    test_info = read_json(test_json)

    ref_filtered = []
    raw_filtered = []

    for item in tqdm(test_info["images"]):
        ref_depth = cv2.imread(os.path.join(data_main_path, item["mesh_refined_path"]), cv2.IMREAD_ANYDEPTH)
        raw_depth = cv2.imread(os.path.join(data_main_path, item["mesh_raw_path"]), cv2.IMREAD_ANYDEPTH)
        
        instance_mask = cv2.imread(os.path.join(data_main_path, item["img_path"].replace("raw", "instance_mask")), cv2.IMREAD_ANYDEPTH) > 0
        ref_filtered.append(((ref_depth < 1e-3)*instance_mask).sum() / instance_mask.sum() )
        raw_filtered.append(((raw_depth < 1e-3)*instance_mask).sum() / instance_mask.sum() )
    
    print("ref filtered : {:.3f}".format(np.array(ref_filtered).mean()))
    print("raw filtered : {:.3f}".format(np.array(raw_filtered).mean()))




def check_converge(tb_path, metrics):
    
    pass
        



if __name__ == "__main__":
    # get_filtered_percantage()
    check_converge()