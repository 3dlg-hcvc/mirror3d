import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
import sys
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
import json
import shutil



# ---------------------------------------------------------------------------- #
#                     class Option to print the option list                    #
# ---------------------------------------------------------------------------- #
class Option():
    def __init__(self):
        self.option_fun = dict()
    
    def add_option(self, option_key, option_discription):
        self.option_fun[option_key] = option_discription
    
    def print_option(self):
        print("OPTION : ")
        for index, item in enumerate(self.option_fun.items()):
            print("({}) {:8} : {}".format(index+1, item[0], item[1]))
    
    def is_input_key_valid(self, input_option, annotated_paths):
        key = input_option.split()[0]
        is_valid = False
        for item in self.option_fun.items():
            if key == item[0].split()[0]:
                is_valid = True

        if "back" in input_option:
            try:
                n = int(input_option.split()[1]) - 1
                if n < 0 or n > len(annotated_paths):
                    is_valid = False
            except:
                is_valid = False
        return is_valid
    
# ---------------------------------------------------------------------------- #
#           Plane annotation tool                                              # 
#           (1) anno_env_setup                                                 #
#           (2) anno_plane_update_imgInfo                                      #
#           (3) anno_update_depth_from_imgInfo                                 #
# ---------------------------------------------------------------------------- #
class Plane_annotation_tool():
    
    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None):
        self.data_main_folder = data_main_folder
        assert os.path.exists(data_main_folder), "please input a valid folder path"
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.error_info_path = os.path.join(data_main_folder, "error_img_list.txt")
        if "m3d" not in data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.check_file()
        self.color_img_list = [os.path.join(data_main_folder, "raw", i) for i in os.listdir(os.path.join(data_main_folder, "raw"))]
        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.border_width = border_width
        self.f = f
        if anno_output_folder == None or not os.path.exists(anno_output_folder):
            self.anno_output_folder = data_main_folder
            print("########## NOTE output saved to {}, this may overwrite your current information ############".format(self.anno_output_folder))

    # ------------------------------ save error path ----------------------------- #
    def save_error_raw_name(self, sample_raw_name):
        error_img_list = []
        if os.path.exists(self.error_info_path):
            error_img_list = read_txt(self.error_info_path)
        if sample_raw_name not in error_img_list:
            with open(self.error_info_path, "a") as file:
                    file.write(sample_raw_name)
                    file.write("\n")
    
    # ---------------------------------------------------------------------------- #
    #           check whether files under self.data_main_folder are valid          #
    # ---------------------------------------------------------------------------- #
    def check_file(self):
        data_correct = True
        raw_foler = os.path.join(self.data_main_folder, "raw")

        if self.is_matterport3d:
            depth_folder = os.path.join(self.data_main_folder, "mesh_raw_depth")
        else:
            depth_folder = os.path.join(self.data_main_folder, "hole_raw_depth")

        mask_folder = os.path.join(self.data_main_folder, "instance_mask")

        for file_name in os.listdir(raw_foler):

            raw_file_path = os.path.join(raw_foler, file_name)
            depth_file_path = os.path.join(depth_folder, file_name)
            
            if self.is_matterport3d:
                depth_file_path = rreplace(depth_file_path, "i", "d")
            mask_file_path =  os.path.join(mask_folder, file_name)
            if os.path.exists(mask_file_path) and os.path.exists(raw_file_path) and os.path.exists(depth_file_path):
                continue
            else:
                data_correct = False
                self.save_error_raw_name(file_name)
                print(" path not exisits : {} mask {} raw {} depth {}".format( file_name, os.path.exists(mask_file_path) , os.path.exists(raw_file_path) , os.path.exists(depth_file_path)))
        
        assert data_correct, "sth wrong with data, please check data first"

    
    # ---------------------------------------------------------------------------- #
    #     generate pcd for annotation and initlize plane parameter using ransac    #
    # ---------------------------------------------------------------------------- #
    def anno_env_setup(self):
        import open3d as o3d
        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        os.makedirs(pcd_save_folder, exist_ok=True)
        mirror_border_vis_save_folder = os.path.join(self.anno_output_folder, "border_vis")
        os.makedirs(mirror_border_vis_save_folder, exist_ok=True)
        plane_parameter_save_folder = os.path.join(self.anno_output_folder, "img_info")
        os.makedirs(plane_parameter_save_folder, exist_ok=True)

        for color_img_path in self.color_img_list:

            # --------------------------------- path init -------------------------------- #
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 
            mask_img_path = color_img_path.replace("raw","instance_mask")
            if self.is_matterport3d:
                depth_img_path = rreplace(color_img_path.replace("raw","mesh_raw_depth"), "i", "d")
            else:
                depth_img_path = color_img_path.replace("raw","hole_raw_depth")
            mask = cv2.imread(mask_img_path)

            # -------------- get pcd and masked RGB image for each instance -------------- #
            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                
                instance_tag = "_idx"
                for i in instance_index:
                    instance_tag += "_{}".format(i)
                instance_tag = smaple_name + instance_tag
                pcd_save_path = os.path.join(pcd_save_folder,  "{}.ply".format(instance_tag))
                if os.path.isfile(pcd_save_path):
                    print(pcd_save_path , "exist! continue")
                    continue

                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                mirror_border_mask = cv2.dilate(binary_instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.border_width,self.border_width))) - binary_instance_mask

                # ------------------- save image with masked mirror boreder ------------------ #
                border_mask_vis_image = visulize_mask_one_image(color_img_path, mirror_border_mask)
                border_mask_vis_save_path = os.path.join(mirror_border_vis_save_folder, "{}.jpg".format(instance_tag)) 
                plt.imsave(border_mask_vis_save_path, border_mask_vis_image)
                print("border_mask_vis_save_path : ", os.path.abspath(border_mask_vis_save_path))

                # ---------------- get pcd with refined mirror depth by ransac --------------- #
                pcd, plane_parameter = refine_pcd_by_mirror_border(binary_instance_mask, mirror_border_mask, depth_img_path, color_img_path, self.f)
                one_plane_para_save_path = os.path.join(plane_parameter_save_folder, "{}.json".format(smaple_name))
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_index)
                print("plane_parameter saved to :", os.path.abspath(one_plane_para_save_path))

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))

                    
    # ---------------------------------------------------------------------------- #
    #       check whether mirror plane is correct (verification & adjustment)      #
    # ---------------------------------------------------------------------------- #
    # ----------------------- requirement : open3d 0.10.0 + ---------------------- #
    def anno_plane_update_imgInfo(self):
        import open3d as o3d
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        os.makedirs(anotation_progress_save_folder, exist_ok=True)
        annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()
        while 1:
            if len(path_to_annotate) == 0:
                print("annotation finished ! XD")
                return
            current_pcd_path = path_to_annotate.pop()
            mirror_border_vis_save_folder = os.path.join(self.anno_output_folder, "border_vis")
            masked_image_path = os.path.join(mirror_border_vis_save_folder, "{}.jpg".format(current_pcd_path.split("/")[-1].split(".")[0]))
            print("mirror to annotation : ",masked_image_path)

            pcd = o3d.io.read_point_cloud(current_pcd_path)

            pcd_name = current_pcd_path.split("/")[-1].split(".")[0]
            img_name = pcd_name.split("_idx_")[0]
            instance_id = [int(i) for i in pcd_name.split("_idx_")[1].split("_")]
            if self.is_matterport3d:
                depth_img_path = os.path.join(self.data_main_folder, "mesh_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
            else:
                depth_img_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(img_name))
            color_img_path = os.path.join(self.data_main_folder, "raw","{}.png".format(img_name))
            mask_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(img_name))
            instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)
            plane_parameter = read_json(os.path.join(self.anno_output_folder, "img_info","{}.json".format(img_name)))[pcd_name.split("_idx_")[1]]["plane_parameter"]
            mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
            mirror_pcd = o3d.geometry.PointCloud()
            mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
            mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
            mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox)
            o3d.visualization.draw_geometries([pcd, mirror_plane])



            option_list = Option()
            option_list.add_option("t", "TRUE : initial plane parameter is correct")
            option_list.add_option("w", "WASTE : sample have error, can not be used (e.g. point cloud too noisy)")
            option_list.add_option("r", "REFINE : need to refine the plane parameter")
            option_list.add_option("back n", "BACK : return n times (e.g. back 3 : give up the recent 3 annotated sample and go back)")
            option_list.add_option("exit", "EXIT : save and exit")
            option_list.print_option()

            input_option = input()

            if not option_list.is_input_key_valid(input_option, annotated_paths):
                print("invalid input, please input again :D")
                continue

            if input_option == "t":
                correct_list.append(current_pcd_path)
                self.save_progress(error_list, correct_list)
                annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()
            elif input_option == "w":
                error_list.append(current_pcd_path)
                self.save_progress(error_list, correct_list)
                annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()
            elif input_option == "exit":
                
                self.save_progress(error_list, correct_list)
                annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()
                print("current progress {} / {}".format(len(annotated_paths), len(annotated_paths) + len(path_to_annotate)))
                exit(1)
            elif "back" in input_option:
                back_path_list = []
                n = int(input_option.split()[1]) - 1
                for i in range(n):
                    back_path_list.append(annotated_paths.pop())
                correct_list = list(set(correct_list) - set(back_path_list))
                error_list = list(set(error_list) - set(back_path_list))

                self.save_progress(error_list, correct_list)
                annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()

            elif input_option == "r":
                
                init_step_size = ((np.max(np.array(pcd.points)[:,0])) - (np.min(np.array(pcd.points)[:,0])))/300
                [p1, p2, p3] = get_picked_points(pcd)
                plane_parameter = get_parameter_from_plane_adjustment(pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)

                one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "img_info"), "{}.json".format(img_name))
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_id)

                correct_list.append(current_pcd_path)
                self.save_progress(error_list, correct_list)
                annotated_paths, path_to_annotate , error_list, correct_list = self.get_progress()
    
    # ----------------------------------------------------------------------------------------------------------- #
    # After plane annotation, update hole_raw_depth / mesh_raw_depth to hole_refined_depth / mesh_refined_depth   #
    # ----------------------------------------------------------------------------------------------------------- #
    def anno_update_depth_from_imgInfo(self):
        raw_image_save_folder = os.path.join(self.data_main_folder, "raw")
        img_info_save_folder = os.path.join(self.anno_output_folder, "img_info")
        for color_img_path in self.color_img_list:
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 
            one_info_file_path = os.path.join(img_info_save_folder, "{}.json".format(smaple_name))
            info = read_json(one_info_file_path)
            
            for one_info in info.items():
                instance_index = [int(i) for i in one_info[0].split("_")]

                mask_img_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(smaple_name))
                mask = cv2.imread(mask_img_path)
                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                plane_parameter = one_info[1]["plane_parameter"]
                if self.is_matterport3d:
                    depth_file_name = "{}.png".format(rreplace(smaple_name,"i","d"))
                    # refine mesh raw depth (only Matterport3d have mesh raw depth)
                    mesh_raw_depth_path = os.path.join(self.data_main_folder, "mesh_raw_depth",depth_file_name)
                    mesh_refined_depth_path = os.path.join(self.data_main_folder, "mesh_refined_depth",depth_file_name)
                    os.makedirs(os.path.split(mesh_refined_depth_path)[0], exist_ok=True)
                    cv2.imwrite(mesh_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(hole_raw_depth_path,cv2.IMREAD_ANYDEPTH),self.f))
                    print("update depth {} {}".format(mesh_refined_depth_path))
                else:
                    depth_file_name = "{}.png".format(smaple_name)
                # refine hole raw depth
                hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth",depth_file_name)
                hole_refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth",depth_file_name)
                os.makedirs(os.path.split(hole_refined_depth_path)[0], exist_ok=True)
                cv2.imwrite(hole_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(hole_raw_depth_path,cv2.IMREAD_ANYDEPTH),self.f))
                print("update depth {}".format(hole_refined_depth_path))

    # ---------------------------------------------------------------------------- #
    #                           save annotation progress                           #
    # ---------------------------------------------------------------------------- #
    def save_progress(self, error_list, correct_list):
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        error_txt_path = os.path.join(anotation_progress_save_folder, "error_list.txt")
        correct_txt_path = os.path.join(anotation_progress_save_folder, "correct_list.txt")
        save_txt(error_txt_path, error_list)
        save_txt(correct_txt_path, correct_list)

    # ---------------------------------------------------------------------------- #
    #                            get annotation progress                           #
    # ---------------------------------------------------------------------------- #
    def get_progress(self):
        pcd_path_list = []
        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        for pcd_name in os.listdir(pcd_save_folder):
            pcd_path_list.append(os.path.join(pcd_save_folder, pcd_name))
            

        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")

        error_txt = os.path.join(anotation_progress_save_folder, "error_list.txt")
        correct_txt = os.path.join(anotation_progress_save_folder, "correct_list.txt")

        if os.path.exists(error_txt):
            error_list = read_txt(error_txt)
        else:
            error_list = []

        if os.path.exists(correct_txt):
            correct_list = read_txt(correct_txt)
        else:
            correct_list = []

        path_to_annotate = []
        annotated_paths = []
        path_to_annotate = list(set(pcd_path_list) - set(error_list) - set(correct_list))
        annotated_paths = list(set(pcd_path_list) - set(path_to_annotate))
        if len(path_to_annotate) > 0:
            path_to_annotate.sort()
        if len(annotated_paths) > 0:
            annotated_paths.sort()
        return annotated_paths, path_to_annotate, error_list, correct_list



class Data_post_processing(Plane_annotation_tool):


    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None, expand_range=100):
        self.data_main_folder = data_main_folder
        assert os.path.exists(data_main_folder), "please input a valid folder path"
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.error_info_path = os.path.join(data_main_folder, "error_img_list.txt")
        if "m3d" not in data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.check_file()
        self.color_img_list = [os.path.join(data_main_folder, "raw", i) for i in os.listdir(os.path.join(data_main_folder, "raw"))]
        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.border_width = border_width
        self.f = f
        if anno_output_folder == None or not os.path.exists(anno_output_folder):
            self.anno_output_folder = data_main_folder
            print("########## NOTE output saved to {}, this may overwrite your current information ############".format(self.anno_output_folder))
        self.expand_range = expand_range

    # ---------------------------------------------------------------------------- #
    #                          clamp data based on 3D bbox                         #
    #           all data should be saved in self.data_main_folder                  #
    # ---------------------------------------------------------------------------- #
    def data_clamping(self):
        import open3d as o3d
        raw_image_save_folder = os.path.join(self.data_main_folder, "raw")
        img_info_save_folder = os.path.join(self.data_main_folder, "img_info")
        for color_img_path in self.color_img_list:
            img_name = os.path.split(color_img_path)[1].split(".")[0] 
            one_info_file_path = os.path.join(img_info_save_folder, "{}.json".format(img_name))
            info = read_json(one_info_file_path)
            
            for one_info in info.items():
                instance_index_str = one_info[0].split("_")
                instance_index_tuple = [int(i) for i in instance_index_str]

                # get mask_img_path, depth_img_path, color_img_path
                mask_img_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(img_name))
                if self.is_matterport3d:
                    depth_img_path = os.path.join(self.data_main_folder, "mesh_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
                else:
                    depth_img_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(img_name))
                depth_file_name = depth_img_path.split("/")[-1]
                color_img_path = os.path.join(self.data_main_folder, "raw","{}.png".format(img_name))
                
                # get mirror_border_mask
                instance_mask = get_grayscale_instanceMask(cv2.imread(mask_img_path), instance_index_tuple)
                mirror_border_mask = cv2.dilate(instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.border_width,self.border_width))) - instance_mask
                # get mirror_bbox
                mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
                 
                if self.is_matterport3d:
                    # refine mesh raw depth (only Matterport3d have mesh raw depth)
                    mesh_refined_depth_path = os.path.join(self.data_main_folder, "mesh_refined_depth", depth_file_name)
                    cv2.imwrite(mesh_refined_depth_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=mesh_refined_depth_path, f=self.f, mirror_border_mask=mirror_border_mask , plane_parameter=one_info[1]["plane_parameter"], expand_range = self.expand_range))
                    print("update depth {} {}".format(mesh_refined_depth_path))

                # refine hole raw depth
                hole_refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth", depth_file_name)
                cv2.imwrite(hole_refined_depth_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=hole_refined_depth_path, f=self.f, mirror_border_mask=mirror_border_mask ,plane_parameter=one_info[1]["plane_parameter"], expand_range = self.expand_range))
                print("update depth {}".format(hole_refined_depth_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--data_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
    parser.add_argument(
        '--index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    args = parser.parse_args()
    
    # plane_anno_tool = Plane_annotation_tool(args.data_main_folder, args.index, False)
    # plane_anno_tool.anno_update_depth_from_imgInfo()
    plane_anno_tool = Data_post_processing(args.data_main_folder, args.index, False)
    plane_anno_tool.data_clamping()