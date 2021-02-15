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
    img_info[str(sample_id)]["plane_parameter"] = plane_parameter
    img_info[str(sample_id)]["mirror_normal"] = plane_parameter[:-1]
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
    """
    plt.figure()
    fig = plt.imshow(depth_img, cmap=plt.get_cmap("magma"))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    figure = plt.gcf()
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=100)