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

if __name__ == "__main__":
    pass