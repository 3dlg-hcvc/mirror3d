import os
import cv2
import argparse
from PIL import Image
import time
import numpy as np
import json
import pathlib
import torch
from utils.pycococreatortools import create_annotation_info
from tqdm import tqdm
import random
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
from annotation.plane_annotation_tool.plane_annotation_tool import Plane_annotation_tool

class Input_Generator(Plane_annotation_tool):

    def __init__(self, mirror_data_main_folder, no_mirror_data_main_folder="", dataset_main_folder="", json_output_folder="", split="test", anchor_normal_path="", contain_no_mirror=False, split_info_folder="", max_num=0):
        """
        Args:
            mirror_data_main_folder : folder that contains "raw" , "instance_mask", "refined_depth" ... folders
            no_mirror_data_main_folder : folder of the original datase
                                           e.g. For Matterport3d dataset, no_mirror_data_main_folder should be the folder path 
                                                that contains "undistorted_color_images", "undistorted_depth_images" folder
            split : "trian / test / val"

        """
        self.split = split
        self.max_num = max_num
        self.no_mirror_data_main_folder = no_mirror_data_main_folder
        self.dataset_main_folder = dataset_main_folder
        self.split_info_folder = split_info_folder
        self.mirror_data_main_folder = mirror_data_main_folder
        if not os.path.exists(json_output_folder):
            self.json_output_folder = os.path.join(self.mirror_data_main_folder, "network_input_json")
            os.makedirs(self.json_output_folder, exist_ok=True)
            print("output json saved to : {}".format(self.json_output_folder))
        else:
            self.json_output_folder = json_output_folder
        self.anchor_normal_path = anchor_normal_path
        
        self.contain_no_mirror = contain_no_mirror
        if self.contain_no_mirror:
            assert os.path.exists(self.no_mirror_data_main_folder), "please input a valid none mirror (original data) main folder"
       
        if "m3d" in self.mirror_data_main_folder:
            self.dataset_name = "m3d"
        elif "nyu" in self.mirror_data_main_folder:
            self.dataset_name = "nyu"
        else:
            self.dataset_name = "scannet"

    def set_split(self, split):
        """
        Set split (train / test/ val)
        """
        self.split = split

    def generate_coco_main(self):

        mirror_colorImg_list = []
        no_mirror_colorImg_list = []

        # Get the mirror color image path list
        split_info_path = os.path.join(self.split_info_folder, "{}_mirror".format(self.dataset_name), "{}.txt".format(self.split))
        mirror_split_id_list = read_txt(split_info_path)
        mirror_color_folder = os.path.join(self.mirror_data_main_folder, "raw")
        for img_name in os.listdir(mirror_color_folder):
            if self.dataset_name == "scannet":
                img_tag = img_name.rsplit("_",1)[0]
            else:
                img_tag = img_name.split(".")[0]
            if img_tag in mirror_split_id_list:
                mirror_colorImg_list.append(os.path.join(mirror_color_folder, img_name))

        # Get the none-mirror color image path list
        if self.contain_no_mirror:
            command = "find {} -type f | grep color".format(self.no_mirror_data_main_folder)
            none_img_file_list = [i.strip() for i in os.popen(command).readlines()]
            split_info_path = os.path.join(self.split_info_folder, "{}_ori".format(self.dataset_name), "{}.txt".format(self.split))
            no_mirror_split_id_list = read_txt(split_info_path)
            for one_img_path in none_img_file_list:
                img_tag = one_img_path.split("/")[-1].split(".")[0]
                img_scene = one_img_path.split("/")[-3]
                if self.dataset_name == "nyu":
                    # NYUv2 data is classified by image_name (with appendix)
                    if img_tag not in mirror_split_id_list and img_tag in no_mirror_split_id_list:
                        no_mirror_colorImg_list.append(one_img_path)
                else:
                    # Matterport3d and ScanNet are classified by sceneID
                    if img_tag not in mirror_split_id_list and img_scene in no_mirror_split_id_list:
                        no_mirror_colorImg_list.append(one_img_path)
        normal_num = np.load(self.anchor_normal_path).shape[0]

        if self.contain_no_mirror:
            cocoFormat_save_path = os.path.join(self.json_output_folder , "{}_{}_normal_all.json".format(self.split, normal_num))
        else:
            cocoFormat_save_path = os.path.join(self.json_output_folder , "{}_{}_normal_mirror.json".format(self.split, normal_num))
        self.colorPath_2_json_only_detection(mirror_colorImg_list, no_mirror_colorImg_list, cocoFormat_save_path)


    def get_anchor_norma_info(self, mirror_normal):
        """
        Args:
            mirror_normal : 1*3 normal
        Output: 
            anchor_normal_class : int
            anchor_normal_residual : list[] ; len = 3
        """
        anchor_normal = np.load(self.anchor_normal_path)

        cloest_distance = 100 # init to be a large number
        for i in range(len(anchor_normal)):
            distance_anchor = mirror_normal - anchor_normal[i] 
            distance = np.sqrt(distance_anchor[0]**2 + distance_anchor[1]**2 + distance_anchor[2]**2)
            if distance < cloest_distance:
                cloest_distance = distance
                anchor_normal_class = i #! the last class is background
                anchor_normal_residual = distance_anchor
        return int(anchor_normal_class), list(anchor_normal_residual)

    

    def colorPath_2_json_only_detection(self, mirror_color_img_list, no_mirror_color_img_list, coco_save_path):
        """
        Args:
            mirror_color_img_list : color image paht list
            coco_save_path : coco format annotation save path
        Output:
            coco format annotation --> saved to coco_save_path
        """     
        raw_folder = os.path.join(self.mirror_data_main_folder, "raw")

        categories_info = dict()
        mirror_label = 1
        categories_info["supercategory"] = "mirror"
        categories_info["id"] = mirror_label
        categories_info["name"] = "mirror"
        
        # COCO annotation id should start from 1
        annotation_id = 1 
        annotations = []
        images = []
        annotation_unique_id = 1
        # Get COCO annoatation for images contain mirror
        for item_index, one_mirror_color_img_path in enumerate(tqdm(mirror_color_img_list)):
            h, w, _ = cv2.imread(one_mirror_color_img_path).shape
            mask_path = one_mirror_color_img_path.replace("raw", "instance_mask")
            img_info_path = one_mirror_color_img_path.replace("raw", "img_info").split(".")[0] + ".json"
            img_info = read_json(img_info_path)
            one_mirror_color_img_path_abv = os.path.relpath(one_mirror_color_img_path, self.dataset_main_folder)
            if not os.path.exists(mask_path) or not os.path.exists(one_mirror_color_img_path):
                continue

            raw_img_path_abv = os.path.relpath(one_mirror_color_img_path, self.dataset_main_folder)

            if self.dataset_name == "m3d":
                mesh_raw_path_abv = rreplace(raw_img_path_abv.replace("raw", "mesh_raw_depth"),"i","d").replace(".jpg", ".png")
                mesh_refined_path_abv = rreplace(raw_img_path_abv.replace("raw", "mesh_refined_depth"),"i","d").replace(".jpg", ".png")
                hole_raw_path_abv = rreplace(raw_img_path_abv.replace("raw", "hole_raw_depth"),"i","d").replace(".jpg", ".png")
                hole_refined_path_abv = rreplace(raw_img_path_abv.replace("raw", "hole_refined_depth"),"i","d").replace(".jpg", ".png")
            else:
                hole_raw_path_abv = raw_img_path_abv.replace("raw", "hole_raw_depth").replace(".jpg", ".png")
                hole_refined_path_abv = raw_img_path_abv.replace("raw", "hole_refined_depth").replace(".jpg", ".png")
                mesh_raw_path_abv = hole_raw_path_abv
                mesh_refined_path_abv = hole_refined_path_abv

            # COCO image[]
            image = {
                    "id": item_index+1, # same as "image_id"
                    "height": h,
                    "width" : w,
                    "img_path": raw_img_path_abv,
                    "mesh_raw_path": mesh_raw_path_abv,
                    "mesh_refined_path": mesh_refined_path_abv,
                    "hole_raw_path": hole_raw_path_abv,
                    "hole_refined_path": hole_refined_path_abv,
                }
            
            # COCO annotation[]
            mask_img = cv2.imread(mask_path)
            mask_path_abv = os.path.relpath(mask_path, self.dataset_main_folder)
            for instance_index in np.unique(np.reshape(mask_img,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                instance_tag = "{}_{}_{}".format(instance_index[0], instance_index[1], instance_index[2])
                ground_truth_binary_mask = get_grayscale_instanceMask(mask_img, instance_index)
                category_info = {'id': mirror_label, 'is_crowd': 0}
                annotation = create_annotation_info(
                                annotation_id, annotation_unique_id, category_info, ground_truth_binary_mask,
                                (w,h), tolerance=2)
                annotation["mirror_normal_camera"] = unit_vector(img_info[str(instance_tag)]["mirror_normal"]).tolist()
                anchor_normal_class, anchor_normal_residual = self.get_anchor_norma_info(annotation["mirror_normal_camera"])
                annotation["anchor_normal_class"] = anchor_normal_class
                annotation["anchor_normal_residual"] = anchor_normal_residual
                annotation["depth_path"] = mesh_refined_path_abv
                annotation["mesh_refined_path"] = mesh_refined_path_abv
                annotation["hole_refined_path"] = hole_refined_path_abv
                annotation["mesh_raw_path"] = mesh_raw_path_abv
                annotation["hole_raw_path"] = hole_raw_path_abv
                annotation["image_path"] = raw_img_path_abv
                annotation["instance_tag"] = str(instance_tag)
                annotation["mask_path"] = mask_path_abv
                annotations.append(annotation)
                annotation_id += 1

            if image not in images:
                images.append(image)
            annotation_unique_id += 1

        # Get COCO annoatation for images don't have mirror
        random.shuffle(no_mirror_color_img_list)
        for one_no_mirror_color_img_path in tqdm(no_mirror_color_img_list):
            raw_img_path_abv = os.path.relpath(one_no_mirror_color_img_path, self.dataset_main_folder)

            if self.dataset_name == "m3d":
                hole_raw_path_abv = rreplace(raw_img_path_abv.replace("color", "depth"),"i","d").replace(".jpg", ".png")
                hole_refined_path_abv = hole_raw_path_abv
                mesh_raw_path_abv = ((rreplace(hole_raw_path_abv, "undistorted_depth_images", "mesh_images")).replace("undistorted_depth_images", "matterport_render_depth")).split(".")[0] + "_mesh_depth.png"
                mesh_refined_path_abv = mesh_raw_path_abv
            else:
                hole_raw_path_abv = raw_img_path_abv.replace("color", "depth").replace(".jpg", ".png")
                hole_refined_path_abv = hole_raw_path_abv
                mesh_raw_path_abv = hole_raw_path_abv
                mesh_refined_path_abv = hole_raw_path_abv

            # COCO image[]
            image = {
                    "id": item_index+1, # same as "image_id"
                    "height": h,
                    "width" : w,
                    "img_path": raw_img_path_abv,
                    "mesh_raw_path": mesh_raw_path_abv,
                    "mesh_refined_path": mesh_refined_path_abv,
                    "hole_raw_path": hole_raw_path_abv,
                    "hole_refined_path": hole_refined_path_abv,
                }

            if image not in images:
                images.append(image)
                item_index+= 1
            
            if self.max_num != 0 and (item_index - annotation_unique_id > self.max_num):
                break 

        coco_format_output = dict()
        coco_format_output["annotations"] = annotations
        coco_format_output["images"] = images
        coco_format_output["info"] = info = {
                                "description": self.mirror_data_main_folder,
                                "date_created": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                            }
        coco_format_output["categories"] = [categories_info]
        save_json(coco_save_path, coco_format_output)  
    


def get_kmeans_normal(train_coco_json="", num_clusters=10, output_save_folder=""):

    from collections import Counter
    from kmeans_pytorch import kmeans


    def anchor_normal_visulization(AN_count,kmeans_normal, save_path):

        color_list  = ["orange","purple","pink","yellow","cyan","gray","lime","tan","gold","salmon","plum","peru","teal","olive"] 
        # plt.cla()
        fig = plt.figure()
        title = ""

        ax = fig.gca(projection='3d')
        ax.set_title(title)
        ax.set_xlim3d(-2000, 2000)
        ax.set_ylim3d(-2000, 2000)
        ax.set_zlim3d(-2000, 2000)
        ax.quiver(0, 0, 0, 0, 0, 1, length = 2000,  color='b', arrow_length_ratio=0.05) # Z
        ax.text(0, 0, 2000, "z", color='b')
        ax.quiver(0, 0, 0, 0, 1, 0, length = 2000,  color='g', arrow_length_ratio=0.05) # Y
        ax.text(0, 2000, 0, "y", color='g')
        ax.quiver(0, 0, 0, 1, 0, 0, length = 2000,  color='r', arrow_length_ratio=0.05) # X
        ax.text(2000, 0, 0, "x", color='r')


        for anchor_id, one_normal in enumerate(kmeans_normal):
            # (X, Z, Y)
            ax.quiver(0, 0, 0, one_normal[0]*1, one_normal[1]*1, one_normal[2]*1, length = 2000,  color=color_list[anchor_id], arrow_length_ratio=0.05) # X
            ax.text(one_normal[0]*2000, one_normal[1]*2000, one_normal[2]*2000, "{}:{}".format(anchor_id,AN_count[anchor_id]), color=color_list[anchor_id], fontsize=8)

        ax.view_init(-80, -90)
        # plt.show()
        plt.savefig(save_path)
        print("normal visulization saved to : ", save_path)

    mirror_normal_list = []
    print("normal generated based on : ", train_coco_json)
    with open(train_coco_json, 'r') as j:
        coco_annotation = json.loads(j.read())

    for info in coco_annotation["annotations"]:
        mirror_normal_list.append(info["mirror_normal_camera"])

    mirror_normal_list = torch.from_numpy(np.array(mirror_normal_list))

    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=mirror_normal_list, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )

    print(cluster_ids_x, cluster_centers)
    kmeans_normal = cluster_centers.numpy()

    os.makedirs(output_save_folder, exist_ok=True)
    npy_save_path = os.path.join(output_save_folder, "m3d_kmeans_normal_{}.npy".format(str(num_clusters)))

    np.save(npy_save_path, kmeans_normal)
    print("numpy saved to : ", npy_save_path)
    AN_count = Counter(cluster_ids_x.tolist())
    anchor_normal_visulization(AN_count, kmeans_normal, npy_save_path.replace("npy","jpg"))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    # Args for --stage 1
    parser.add_argument(
        '--mirror_data_main_folder', default="", type=str, help="folder contain raw, instance_mask... folders") 
    parser.add_argument(
        '--stage', default="", type=str, help="(1) generate coco format input (2) generate anchor normal") 
    parser.add_argument(
        '--no_mirror_data_main_folder', default="", type=str, help="dataset main folder") 
    parser.add_argument(
        '--dataset_main_folder', default="", type=str, help="output path in .json will be the relative path to --dataset_main_folder") 
    parser.add_argument(
        '--split_info_folder', default="", type=str, help="split_info.zip unzip folder") 
    parser.add_argument(
        '--json_output_folder', default="", type=str, help="dataset main folder") 
    parser.add_argument(
        '--split', default="all", type=str, help="train / test/ val") 
    parser.add_argument('--contain_no_mirror', help='do multi-process or not',action='store_true')
    parser.add_argument(
        '--anchor_normal_path', default="", type=str, help="anchor normal path") 
    parser.add_argument(
        '--max_num', default=0, type=int, help="max number of none-mirror samples")
    # Args for --stage 2
    parser.add_argument(
        '--output_save_folder', default="", type=str, help="kmeans anchor normal saved path")
    parser.add_argument(
        '--coco_json', default="", type=str, help="coco format json file to generate the kmeans normal")
    parser.add_argument(
        '--num_clusters', default=10, type=int, help="number of cluster center")
    args = parser.parse_args()
    generator = Input_Generator(mirror_data_main_folder = args.mirror_data_main_folder, \
                                no_mirror_data_main_folder = args.no_mirror_data_main_folder, \
                                dataset_main_folder = args.dataset_main_folder, \
                                json_output_folder = args.json_output_folder, \
                                split = args.split, \
                                anchor_normal_path = args.anchor_normal_path, \
                                contain_no_mirror = args.contain_no_mirror, \
                                split_info_folder = args.split_info_folder, \
                                max_num = args.max_num)

    if args.stage == "1":
        assert os.path.exists(args.anchor_normal_path), "please input a anchor normal .npy path"
        assert os.path.exists(args.dataset_main_folder), "please input a valid dataset main folder"
        assert os.path.exists(args.mirror_data_main_folder), "please input a valid mirror data main folder"
        assert os.path.exists(args.split_info_folder), "please input a split information folder (please remember to down load the split_info.zip from http://aspis.cmpt.sfu.ca/projects/mirrors/data_release/split_info.zip)" 
        if args.split == "all":
            generator.set_split("train")
            generator.generate_coco_main()
            generator.set_split("test")
            generator.generate_coco_main()
            if args.mirror_data_main_folder.find("nyu") <=0 :
                generator.set_split("val")
                generator.generate_coco_main()
        else:
            generator.generate_coco_main()
    elif args.stage == "2":
        get_kmeans_normal(train_coco_json=args.coco_json, num_clusters=args.num_clusters, output_save_folder=args.output_save_folder)