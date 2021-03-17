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
import time



class Plane_annotation_tool():
    """
    The Plane_annotation_tool class currently does the following:
    1. anno_env_setup          
    2. anno_plane_update_imgInfo                       
    3. anno_update_depth_from_imgInfo          
    """
    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None):
        """
        Initilization

        Args:
            data_main_folder : Folder raw, hole_raw_depth/ mesh_raw_depth, instance_mask saved folder.
            anno_output_folder(optional) : Inital pcd, img_info, border_vis saved forder (default : data_main_folder).
            process_index : The process index of multi_processing.
            multi_processing : Use multi_processing or not (bool).
            border_width : Half of mirror 2D border width (half of cv2.dilate kernel size; 
                           default kernel anchor is at the center); default : 50 --> actualy border width = 25.
            f : Camera focal length of current input data.
        """

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
        # Because we use "cv2.getStructuringElement(cv2.MORPH_ELLIPSE ***" and the anchor of the element is at the center, so actual width is half of self.border_width
        self.border_width = border_width * 2 
        self.f = f
        if anno_output_folder == None or not os.path.exists(anno_output_folder):
            self.anno_output_folder = data_main_folder
            print("########## NOTE output saved to {}, this may overwrite your current information ############".format(self.anno_output_folder))
        else:
            self.anno_output_folder = anno_output_folder
    

    def set_show_plane(self, show_plane):
        """
        For step 2: show the mesh plane during annotation or not
        Suggest to show the mesh plane if computer allows
        """
        self.show_plane = show_plane

    def save_error_raw_name(self, sample_raw_name):
        """
        Save error path
         
        Args : 
            sample_raw_name : color image name
        """

        error_img_list = []
        if os.path.exists(self.error_info_path):
            error_img_list = read_txt(self.error_info_path)
        if sample_raw_name not in error_img_list:
            with open(self.error_info_path, "a") as file:
                    file.write(sample_raw_name)
                    file.write("\n")
    
    def check_file(self):
        """Check whether files under self.data_main_folder are valid"""

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
                print(" path not exisits : {} {} mask {} {}raw {} depth {} {}".format( file_name, os.path.exists(mask_file_path),mask_file_path , os.path.exists(raw_file_path),raw_file_path , os.path.exists(depth_file_path), depth_file_path))
        
        assert data_correct, "sth wrong with data, please check data first"

    

    def anno_env_setup(self):
        """
        Generate pcd for annotation and initlize plane parameter using ransac
        
        Output:
            pcd : .ply file (per instance).
            img_info : .json file (per image); save mirror instances' parameter. 
            border_vis : .png file (per instance).
        """
        import open3d as o3d
        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        os.makedirs(pcd_save_folder, exist_ok=True)
        mirror_border_vis_save_folder = os.path.join(self.anno_output_folder, "border_vis")
        os.makedirs(mirror_border_vis_save_folder, exist_ok=True)
        plane_parameter_save_folder = os.path.join(self.anno_output_folder, "img_info")
        os.makedirs(plane_parameter_save_folder, exist_ok=True)

        for color_img_path in self.color_img_list:
            # Get paths
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 
            mask_img_path = color_img_path.replace("raw","instance_mask")
            if self.is_matterport3d:
                depth_img_path = rreplace(color_img_path.replace("raw","mesh_raw_depth"), "i", "d")
            else:
                depth_img_path = color_img_path.replace("raw","hole_raw_depth")
            mask = cv2.imread(mask_img_path)

            #  Get pcd and masked RGB image for each instance
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

                #  Save image with masked mirror boreder 
                border_mask_vis_image = visulize_mask_one_image(color_img_path, mirror_border_mask)
                border_mask_vis_save_path = os.path.join(mirror_border_vis_save_folder, "{}.jpg".format(instance_tag)) 
                plt.imsave(border_mask_vis_save_path, border_mask_vis_image)
                print("border_mask_vis_save_path : ", os.path.abspath(border_mask_vis_save_path))

                #  Get pcd with refined mirror depth by ransac 
                pcd, plane_parameter = refine_pcd_by_mirror_border(binary_instance_mask, mirror_border_mask, depth_img_path, color_img_path, self.f)
                one_plane_para_save_path = os.path.join(plane_parameter_save_folder, "{}.json".format(smaple_name))
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_index)
                print("plane_parameter saved to :", os.path.abspath(one_plane_para_save_path))

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))


    def anno_plane_update_imgInfo(self):
        """
        Check whether mirror plane is correct (verification & adjustment)

        Requirement : open3d 0.10.0 +
        """
        import open3d as o3d
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        os.makedirs(anotation_progress_save_folder, exist_ok=True)
        self.get_progress() # self.sample_index start from 0
        annotation_start_index = self.sample_index
        annotation_start_time = time.time()
        while 1:
            if self.sample_index == len(self.pcd_path_list):
                print("annotation finished ! XD")
                return
            current_pcd_path = self.pcd_path_list[self.sample_index]
            mirror_border_vis_save_folder = os.path.join(self.anno_output_folder, "border_vis")
            masked_image_path = os.path.join(mirror_border_vis_save_folder, "{}.jpg".format(current_pcd_path.split("/")[-1].split(".")[0]))
            current_sample_status = "N/A"
            if current_pcd_path in self.correct_list:
                current_sample_status = "correct"
            elif current_pcd_path in self.error_list:
                current_sample_status = "error"
            print("###################### sample status {} ######################".format(current_sample_status))
            print("sample index {} mirror to annotate {}".format(self.sample_index, masked_image_path))

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

            if self.show_plane:
                instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)
                plane_parameter = read_json(os.path.join(self.anno_output_folder, "img_info","{}.json".format(img_name)))[pcd_name.split("_idx_")[1]]["plane_parameter"]
                mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))

            if self.show_plane:
                try:
                    mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox)
                    o3d.visualization.draw_geometries([pcd, mirror_plane])
                except:
                    o3d.visualization.draw_geometries([pcd])
            else:
                o3d.visualization.draw_geometries([pcd])

            option_list = Tool_Option()
            option_list.add_option("t", "TRUE : initial plane parameter is correct")
            option_list.add_option("w", "WASTE : sample have error, can not be used (e.g. point cloud too noisy)")
            option_list.add_option("r", "REFINE : need to refine the plane parameter")
            option_list.add_option("back n", "BACK : return n times (e.g. back 3 : give up the recent 3 annotated sample and go back)")
            option_list.add_option("goto n", "GOTO : goto the n th image (e.g. goto 3 : go to the third image")
            option_list.add_option("n", "NEXT : goto next image without annotation")
            option_list.add_option("exit", "EXIT : save and exit")
            option_list.print_option()

            input_option = input()

            if not option_list.is_input_key_valid(input_option):
                print("invalid input, please input again :D")
                continue
            
            if input_option == "t":
                self.correct_list.append(current_pcd_path)
                self.save_progress()
                self.get_progress()
            elif input_option == "w":
                self.error_list.append(current_pcd_path)
                self.save_progress()
                self.get_progress()
            elif input_option == "n":
                if current_sample_status == "N/A":
                    print("please annotate current sample :-)")
                    continue
                self.sample_index += 1
            elif input_option == "exit":
                self.save_progress()
                self.get_progress()
                print("current progress {} / {}".format(self.sample_index, len(self.pcd_path_list)))
                refer_speed = (time.time()-annotation_start_time)/ (self.sample_index - annotation_start_index)
                left_h = ((len(self.pcd_path_list) - self.sample_index) * refer_speed) / 3600
                print("Reference annotation speed {:.2f} s/sample; Estimate remaining time {:.1f} h.".format(refer_speed, left_h))
                exit(1)
            elif "back" in input_option:
                n = int(input_option.split()[1]) - 1
                if self.sample_index - n < 0:
                    print("at most return {} times".format(self.sample_index+1))
                    continue
                self.sample_index -= n
            elif "goto" in input_option:
                n = int(input_option.split()[1]) - 1
                if  n > len(self.pcd_path_list)-1:
                    print("you can go to 0 ~ {}".format(len(self.pcd_path_list)-1))
                    continue
                self.sample_index = n

            elif input_option == "r":
                
                init_step_size = ((np.max(np.array(pcd.points)[:,0])) - (np.min(np.array(pcd.points)[:,0])))/300
                [p1, p2, p3] = get_picked_points(pcd)
                plane_parameter = get_parameter_from_plane_adjustment(pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)

                one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "img_info"), "{}.json".format(img_name))
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_id)

                self.correct_list.append(current_pcd_path)
                self.save_progress()
                self.get_progress()

    def anno_update_depth_from_imgInfo(self):
        """
        After plane annotation, update "hole_raw_depth/mesh_raw_depth" to "hole_refined_depth/mesh_refined_depth"

        Output:
            Refined depth saved to hole_refined_depth or mesh_refined_depth (Matterport3d only).
        """
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

                # Refine mesh raw depth (only Matterport3d have mesh raw depth)
                if self.is_matterport3d:
                    depth_file_name = "{}.png".format(rreplace(smaple_name,"i","d"))
                    mesh_raw_depth_path = os.path.join(self.data_main_folder, "mesh_raw_depth",depth_file_name)
                    mesh_refined_depth_path = os.path.join(self.data_main_folder, "mesh_refined_depth",depth_file_name)
                    os.makedirs(os.path.split(mesh_refined_depth_path)[0], exist_ok=True)
                    # If there's refined depth; refine the refiend depth
                    if os.path.exists(hole_refined_depth_path):
                        hole_raw_depth_path = hole_refined_depth_path
                    cv2.imwrite(mesh_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(hole_raw_depth_path,cv2.IMREAD_ANYDEPTH),self.f))
                    print("update depth {} {}".format(mesh_refined_depth_path))
                else:
                    depth_file_name = "{}.png".format(smaple_name)
                # Refine hole raw depth
                hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth",depth_file_name)
                hole_refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth",depth_file_name)
                os.makedirs(os.path.split(hole_refined_depth_path)[0], exist_ok=True)
                # If there's refined depth; refine the refiend depth
                if os.path.exists(hole_refined_depth_path):
                    hole_raw_depth_path = hole_refined_depth_path
                cv2.imwrite(hole_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(hole_raw_depth_path,cv2.IMREAD_ANYDEPTH),self.f))
                print("update depth {}".format(hole_refined_depth_path))

    def save_progress(self):
        """Save annotation progress"""
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        error_txt_path = os.path.join(anotation_progress_save_folder, "error_list.txt")
        correct_txt_path = os.path.join(anotation_progress_save_folder, "correct_list.txt")
        save_txt(error_txt_path, set(self.error_list))
        save_txt(correct_txt_path, set(self.correct_list))

    def get_progress(self):
        """Get annotation progress"""
        self.sample_index = 0
        start_index = 0
        self.pcd_path_list = []
        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        for pcd_name in os.listdir(pcd_save_folder):
            self.pcd_path_list.append(os.path.join(pcd_save_folder, pcd_name))
        self.pcd_path_list.sort()
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")

        error_txt = os.path.join(anotation_progress_save_folder, "error_list.txt")
        correct_txt = os.path.join(anotation_progress_save_folder, "correct_list.txt")

        if os.path.exists(error_txt):
            self.error_list = read_txt(error_txt)
        else:
            self.error_list = []

        if os.path.exists(correct_txt):
            self.correct_list = read_txt(correct_txt)
        else:
            self.correct_list = []

        for index, one_path in enumerate(self.pcd_path_list):
            if one_path not in self.correct_list and one_path not in self.error_list:
                self.sample_index = index
                return



    def adjust_one_sample_plane(self, instance_index=None, img_name=None):
        """
        Repeatedly adjust one sample's plane parameter

        Args:
            instance_index : "[R]_[G]_[B]",e.g. (128, 0, 0) --> "128_0_0"
            img_name : color image sample name, e.g. 128
        """
        if len(img_name) == 0 or len(instance_index) == 0:
            print("invalid input instance_index {} img_name {}".format(instance_index, img_name))
            exit()

        import open3d as o3d
        if self.is_matterport3d:
            hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
            mesh_raw_depth_path = os.path.join(self.data_main_folder, "mesh_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
            mesh_refined_depth_path = os.path.join(self.anno_output_folder, "mesh_refined_depth","{}.png".format(rreplace(img_name, "i", "d")))
            hole_refined_depth_path = os.path.join(self.anno_output_folder, "hole_refined_depth","{}.png".format(rreplace(img_name, "i", "d")))
        else:
            hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(img_name))
            hole_refined_depth_path = os.path.join(self.anno_output_folder, "hole_refined_depth","{}.png".format(img_name))

        color_img_path = os.path.join(self.data_main_folder, "raw","{}.png".format(img_name))
        mask_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(img_name))
        instance_id = [int(i) for i in instance_index.split("_")]
        instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)

        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        instance_tag = img_name + "_idx_" + instance_index
        pcd_path = os.path.join(pcd_save_folder, "{}.ply".format(instance_tag))

        
        one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "img_info"), "{}.json".format(img_name))
        if os.path.exists(one_plane_para_save_path):
            plane_parameter = read_json(one_plane_para_save_path)[instance_index]["plane_parameter"]

        pcd = o3d.io.read_point_cloud(pcd_path)
        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[0,0,1])
        o3d.visualization.draw_geometries([pcd, mirror_pcd])
        init_step_size = ((np.max(np.array(pcd.points)[:,0])) - (np.min(np.array(pcd.points)[:,0])))/300

        while 1:

            option_list = Tool_Option()
            option_list.add_option("f", "FINISH : update hole_refined_depth/ mesh_refined_depth/ img_info and EXIT")
            option_list.add_option("a", "ADJUST : need to adjust the plane parameter")
            option_list.add_option("i", "INIT : pick 3 points to initialize the plane")
            option_list.print_option()
            input_option = input()

            print("relevant color image path : {}".format(color_img_path))

            if input_option not in ["f", "i", "a"]:
                print("invalid input, please input again :D")
                continue
            
            if input_option == "f":
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_id)
                cv2.imwrite(hole_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, instance_mask, cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH),self.f))
                if self.is_matterport3d:
                    cv2.imwrite(mesh_refined_depth_path, refine_depth_with_plane_parameter_mask(plane_parameter, instance_mask, cv2.imread(mesh_raw_depth_path, cv2.IMREAD_ANYDEPTH),self.f))
                print("annotation of {} finished !".format(img_name))
                exit()
            elif input_option == "i":
                [p1, p2, p3] = get_picked_points(pcd)
                plane_parameter = get_parameter_from_plane_adjustment(pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)
                mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[0,0,1])
                o3d.visualization.draw_geometries([pcd, mirror_pcd])

            elif input_option == "a":
                p1 = np.mean(np.array(mirror_pcd.points), axis=0)
                p2 = np.array(mirror_pcd.points)[0]
                p3 = np.array(mirror_pcd.points)[-1]
                plane_parameter = get_parameter_from_plane_adjustment(pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)
                mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[0,0,1])
                o3d.visualization.draw_geometries([pcd, mirror_pcd])



    def manual_clamp_one_sample(self, instance_index=None, img_name=None):
        """
        Repeatedly adjust one sample's plane parameter

        Args:
            instance_index : "[R]_[G]_[B]",e.g. (128, 0, 0) --> "128_0_0"
            img_name : color image sample name, e.g. 128
        """
        if len(instance_index) == 0 or len(instance_index) == 0:
            print("invalid input instance_index {} img_name {}".format(instance_index, img_name))
            exit()
        clamp_dis = 100

        import open3d as o3d
        if self.is_matterport3d:
            hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
            hole_refined_depth_path = os.path.join(self.anno_output_folder, "hole_refined_depth","{}.png".format(rreplace(img_name, "i", "d")))
        else:
            hole_raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(img_name))
            hole_refined_depth_path = os.path.join(self.anno_output_folder, "hole_refined_depth","{}.png".format(img_name))

        color_img_path = os.path.join(self.data_main_folder, "raw","{}.png".format(img_name))
        mask_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(img_name))
        instance_id = [int(i) for i in instance_index.split("_")]
        instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)

        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        instance_tag = img_name + "_idx_" + instance_index
        pcd_path = os.path.join(pcd_save_folder, "{}.ply".format(instance_tag))

        
        one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "img_info"), "{}.json".format(img_name))
        if os.path.exists(one_plane_para_save_path):
            plane_parameter = read_json(one_plane_para_save_path)[instance_index]["plane_parameter"]

        refined_depth_to_clamp = cv2.imread(hole_refined_depth_path, cv2.IMREAD_ANYDEPTH)
        h, w = refined_depth_to_clamp.shape

        while 1:

            pcd = get_pcd_from_rgb_depthMap(self.f, refined_depth_to_clamp, color_img_path)
            o3d.visualization.draw_geometries([pcd])

            option_list = Tool_Option()
            option_list.add_option("f", "FINISH : update hole_refined_depth/ mesh_refined_depth/ img_info and EXIT")
            option_list.add_option("r", "REPAIR : pick points and refine the specific area")
            option_list.add_option("d", "DISTANCE : the clamping distance_threshold; distance over distance_threshold will be clamped")
            option_list.add_option("exit", "EXIT : exit without saving the result")
            option_list.print_option()
            input_option = input()

            print("relevant color image path : {}".format(color_img_path))

            if input_option not in ["f", "r", "exit", "d"]:
                print("invalid input, please input again :D")
                continue

            if input_option == "d":
                clamp_dis = int(input("please input new clamping distace (default : 100)"))
            elif input_option == "f":
                cv2.imwrite(hole_refined_depth_path, refined_depth_to_clamp)
                print("annotation of {} finished !".format(hole_refined_depth_path))
                exit()
            elif input_option == "r":
                three_points = get_picked_points(pcd)
                three_points_2D = get_2D_coor_from_3D(three_points, self.f, w, h)
                clamp_mask = get_triange_mask(three_points_2D, w, h) 
                refined_depth_to_clamp = clamp_pcd_by_mask(depth_to_refine=refined_depth_to_clamp, f=self.f, clamp_mask=clamp_mask,plane_parameter=plane_parameter, clamp_dis=clamp_dis)
                
            elif input_option == "exit":
                exit()


class Data_post_processing(Plane_annotation_tool):


    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None, expand_range=100, clamp_dis = 100):
        """
        Initilization

        Args:
            data_main_folder : Folder raw, hole_raw_depth/ mesh_raw_depth, instance_mask saved folder.
            anno_output_folder(optional) : Inital pcd, img_info, border_vis saved forder (default : data_main_folder).
            process_index : The process index of multi_processing.
            multi_processing : Use multi_processing or not (bool).
            border_width : Half of mirror 2D border width (half of cv2.dilate kernel size; 
                           default kernel anchor is at the center); default : 50 --> actualy border width = 25.
            f : Camera focal length of current input data.
            expand_range : Data clamping 3D bbox expanded step size.
        """
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
        self.clamp_dis = clamp_dis


    def data_clamping(self):
        """
        Clamp data based on 3D bbox

        Output:
            Clamped depth : saved to hole_refined_depth or mesh_refined depth under self.data_main_folder
        """
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

                # Get mask_img_path, depth_img_path, color_img_path
                mask_img_path = os.path.join(self.data_main_folder, "instance_mask","{}.png".format(img_name))
                if self.is_matterport3d:
                    depth_img_path = os.path.join(self.data_main_folder, "mesh_raw_depth","{}.png".format(rreplace(img_name, "i", "d")))
                else:
                    depth_img_path = os.path.join(self.data_main_folder, "hole_raw_depth","{}.png".format(img_name))
                depth_file_name = depth_img_path.split("/")[-1]
                color_img_path = os.path.join(self.data_main_folder, "raw","{}.png".format(img_name))
                
                # Get mirror_border_mask
                instance_mask = get_grayscale_instanceMask(cv2.imread(mask_img_path), instance_index_tuple)
                mirror_border_mask = cv2.dilate(instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.border_width,self.border_width))) - cv2.erode(instance_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
                
                # Get mirror_bbox
                mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
                 
                if self.is_matterport3d:
                    # Refine mesh raw depth (only Matterport3d have mesh raw depth)
                    mesh_refined_depth_path = os.path.join(self.data_main_folder, "mesh_refined_depth", depth_file_name)
                    cv2.imwrite(mesh_refined_depth_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=mesh_refined_depth_path, f=self.f, mirror_border_mask=mirror_border_mask , plane_parameter=one_info[1]["plane_parameter"], expand_range = self.expand_range, clamp_dis = self.clamp_dis))
                    print("update depth {} {}".format(mesh_refined_depth_path))

                # Refine hole raw depth
                hole_refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth", depth_file_name)
                cv2.imwrite(hole_refined_depth_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=hole_refined_depth_path, f=self.f, mirror_border_mask=mirror_border_mask ,plane_parameter=one_info[1]["plane_parameter"], expand_range = self.expand_range, clamp_dis = self.clamp_dis))
                print("update depth {}".format(hole_refined_depth_path))

    def update_imgInfo_based_on_depth(self):
        """
        Updata img_info based on refined depth

        Output:
            updated img_info : (1) Matterport3d's img_info are updated based on mesh_refined_depth
                               (2) Other datasets img_info are updated based on hole_refined_depth
        """
        for color_img_path in self.color_img_list:
            img_name = os.path.split(color_img_path)[1].split(".")[0]
            if self.is_matterport3d:
                depth_img_path = os.path.join(self.data_main_folder, "mesh_refined_depth","{}.png".format(rreplace(img_name, "i", "d")))
            else:
                depth_img_path = os.path.join(self.data_main_folder, "hole_refined_depth","{}.png".format(img_name))
            mask_img_path = color_img_path.replace("raw", "instance_mask")
            img_info_path = color_img_path.replace("raw", "img_info").replace(".png",".json")
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 

            mask = cv2.imread(mask_img_path)

            
            #  Get plane parameter for each instance (based on refined depth) 
            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                
                instance_tag = "_idx"
                for i in instance_index:
                    instance_tag += "_{}".format(i)
                instance_tag = smaple_name + instance_tag

                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                
                plane_parameter = get_mirror_parameter_from_xyzs_by_ransac(get_points_in_mask(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask))
                save_plane_parameter_2_json(plane_parameter, img_info_path, instance_index)
                print("updated plane_parameter in {}".format(img_info_path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="4")
    parser.add_argument(
        '--data_main_folder', default="")
    parser.add_argument(
        '--process_index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    parser.add_argument('--anno_show_plane', help='do multi-process or not',action='store_true')
    parser.add_argument(
        '--border_width', default=50, type=int, help="border_width")
    parser.add_argument(
        '--f', default=519, type=int, help="camera focal length")
    parser.add_argument(
        '--expand_range', default=200, type=int, help="expand the mirror instance bbox by expand_range; unit : mm")
    parser.add_argument(
        '--clamp_dis', default=100, type=int, help="outliers threshold")
    parser.add_argument(
        '--anno_output_folder', default="")
    parser.add_argument(
        '--img_name', default="")
    parser.add_argument(
        '--instance_index', default="")
    args = parser.parse_args()


    # data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None
    if args.stage == "1":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.anno_env_setup()
    elif args.stage == "2":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.set_show_plane(args.anno_show_plane)
        plane_anno_tool.anno_plane_update_imgInfo()
    elif args.stage == "3":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.anno_update_depth_from_imgInfo()
    elif args.stage == "4": 
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder, expand_range=args.expand_range, clamp_dis=args.clamp_dis)
        plane_anno_tool.data_clamping()
    elif args.stage == "all":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.anno_env_setup()
        plane_anno_tool.anno_plane_update_imgInfo()
        plane_anno_tool.anno_update_depth_from_imgInfo()
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder, expand_range=args.expand_range, clamp_dis=args.clamp_dis)
        plane_anno_tool.data_clamping()
    elif args.stage == "5":
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.update_imgInfo_based_on_depth()
    elif args.stage == "6":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.adjust_one_sample_plane(args.instance_index, args.img_name)
    elif args.stage == "7":
        plane_anno_tool = Plane_annotation_tool(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.manual_clamp_one_sample(args.instance_index, args.img_name)
        