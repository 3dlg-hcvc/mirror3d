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
from PIL import ImageColor


class Plane_annotation_tool():
    """
    The Plane_annotation_tool class currently does the following:
    1. anno_env_setup          
    2. anno_plane_update_imgInfo                       
    3. anno_update_depth_from_imgInfo          
    """
    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, border_width=50, f=519, anno_output_folder=None, mask_version="precise"):
        """
        Initilizationse

        Args:
            data_main_folder : Folder raw, raw_sensorD/ raw_meshD, instance_mask saved folder.
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
        self.mask_version = mask_version
        if "mp3d" not in data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True

        self.color_img_list = [i.strip() for i in os.popen("find -L {} -type f".format(os.path.join(data_main_folder, "mirror_color_images"))).readlines()]
        
        self.color_img_list.sort()
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
    
    def set_color_list(self, color_img_list_path):
        self.color_img_list =list( set(read_txt(color_img_list_path)))
        self.color_img_list.sort()
        if self.multi_processing:
            self.color_img_list = self.color_img_list[self.process_index:self.process_index+1]

    def set_show_plane(self, show_plane):
        """
        For step 2: show the mesh plane during annotation or not
        Suggest to show the mesh plane if computer allows
        """
        self.show_plane = show_plane
    

    def set_overwrite(self, overwrite):
        """
        For STEP 1 overwrite current environment setup or not
        """
        self.overwrite = overwrite

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
        raw_foler = os.path.join(self.data_main_folder, "mirror_color_images")

        if self.is_matterport3d:
            depth_folder = os.path.join(self.data_main_folder, "raw_meshD")
        else:
            depth_folder = os.path.join(self.data_main_folder, "raw_sensorD")

        mask_folder = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version))
        img_info_folder = os.path.join(self.data_main_folder, "mirror_plane")

        for file_name in os.listdir(raw_foler):

            raw_file_path = os.path.join(raw_foler, file_name)
            depth_file_path = os.path.join(depth_folder, file_name).replace("jpg","png")
            
            if self.is_matterport3d:
                depth_file_path = rreplace(depth_file_path, "i", "d").replace("jpg","png")
            mask_file_path =  os.path.join(mask_folder, file_name).replace("jpg","png")
            if os.path.exists(mask_file_path) and os.path.exists(raw_file_path) and os.path.exists(depth_file_path):
                continue
            else:
                data_correct = False
                self.save_error_raw_name(file_name)
                print(" path not exists : {} {} mask {} {}raw {} depth {} {}".format( file_name, os.path.exists(mask_file_path),mask_file_path , os.path.exists(raw_file_path),raw_file_path , os.path.exists(depth_file_path), depth_file_path, os.path.exists(img_info_file_path)))
        
        assert data_correct, "sth wrong with data, please check data first"

        print("dataset checking finished ~ ")
    


    def anno_env_setup(self):
        """
        Generate pcd for annotation and initlize plane parameter using ransac
        
        Output:
            pcd : .ply file (per instance).
            img_info : .json file (per image); save mirror instances' parameter. 
            border_vis : .png file (per instance).
        """
        import open3d as o3d

        def gen_pcd(color_img_path, depth_img_path, mask_img_path):
            #  Get pcd and masked RGB image for each instance
            pcd_save_folder = os.path.split(color_img_path.replace(self.data_main_folder, self.anno_output_folder).replace("mirror_color_images", "anno_pcd"))[0]
            os.makedirs(pcd_save_folder, exist_ok=True)
            mirror_border_vis_save_folder = pcd_save_folder.replace("anno_pcd", "border_vis")
            os.makedirs(mirror_border_vis_save_folder, exist_ok=True)
            plane_parameter_save_folder = pcd_save_folder.replace("anno_pcd", "mirror_plane")
            os.makedirs(plane_parameter_save_folder, exist_ok=True)
            mask = cv2.imread(mask_img_path)
            import pdb;pdb.set_trace()
            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                instance_tag = "_idx_"
                instance_tag += '%02x%02x%02x' % (instance_index[0],instance_index[1],instance_index[2]) # BGR
                smaple_name = os.path.split(color_img_path)[-1].split(".")[0]
                instance_tag = smaple_name + instance_tag
                pcd_save_path = os.path.join(pcd_save_folder,  "{}.ply".format(instance_tag))

                if os.path.isfile(pcd_save_path) and not self.overwrite:
                    print(pcd_save_path , "exist! continue")
                    continue
                else:
                    if os.path.exists(pcd_save_path):
                        print("begin to overwrite {}".format(pcd_save_path))
                    else:
                        print("generating pcd {}".format(pcd_save_path))

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

        for color_img_path in self.color_img_list:
            # Get paths
            mask_img_path = color_img_path.replace("mirror_color_images","mirror_instance_mask_{}".format(self.mask_version)).replace("jpg","png")
            if self.is_matterport3d:
                depth_img_path = rreplace(color_img_path.replace("mirror_color_images","raw_meshD"), "i", "d").replace("jpg","png")
            else:
                depth_img_path = color_img_path.replace("mirror_color_images","raw_sensorD").replace("jpg","png")
            gen_pcd(color_img_path, depth_img_path, mask_img_path)

    def anno_plane_update_imgInfo(self):
        """
        Check whether mirror plane is correct (verification & adjustment)

        Requirement : open3d 0.10.0 +
        """
        import open3d as o3d
        import warnings
        self.check_file()
        warnings.filterwarnings("ignore")
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        os.makedirs(anotation_progress_save_folder, exist_ok=True)
        self.get_progress() # self.sample_index start from 0
        annotation_start_index = self.sample_index
        manual_adjust_num = 0
        annotation_start_time = time.time()
        while 1:
            if self.sample_index == len(self.pcd_path_list):
                print("annotation finished ! XD")
                exit(1)
            current_pcd_path = self.pcd_path_list[self.sample_index]
            currect_pcd_id = current_pcd_path.split("/")[-1].split("_idx_")[0]

            # If one instance in the sample is negative; then this sample is invalid
            if currect_pcd_id in self.error_id: 
                self.error_list.append(current_pcd_path)
                self.save_progress()
                self.get_progress()
                print("[AUTO] sample index {} path {} is invalid".format(self.sample_index, current_pcd_path))
                continue

            mirror_border_vis_save_folder = os.path.join(self.anno_output_folder, "border_vis")
            masked_image_path = os.path.join(mirror_border_vis_save_folder, "{}.jpg".format(current_pcd_path.split("/")[-1].split(".")[0]))
            current_sample_status = "N/A"
            if current_pcd_path in self.correct_list:
                current_sample_status = "correct"
            elif current_pcd_path in self.error_list:
                current_sample_status = "error"
            print("###################### sample status {} ######################".format(current_sample_status))

            

            pcd = o3d.io.read_point_cloud(current_pcd_path)
            pcd_name = current_pcd_path.split("/")[-1].split(".")[0]
            img_name = pcd_name.split("_idx_")[0]
            instance_id = pcd_name.split("_idx_")[1]
            instance_id = ImageColor.getcolor("#{}".format(instance_id), "RGB")
            instance_id = [instance_id[2], instance_id[1], instance_id[0]]# BGR
            if self.is_matterport3d:
                depth_img_path = os.path.join(self.data_main_folder, "raw_meshD","{}.png".format(rreplace(img_name, "i", "d")))
            else:
                depth_img_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(img_name))
            color_img_path = os.path.join(self.data_main_folder, "mirror_color_images","{}.jpg".format(img_name))
            mask_path = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version),"{}.png".format(img_name))
            plane_parameter = read_plane_json(os.path.join(self.anno_output_folder, "mirror_plane","{}.json".format(img_name)))[pcd_name.split("_idx_")[1]]["plane_parameter"]

            if os.path.exists(masked_image_path):
                print("sample index {} mirror to annotate {}".format(self.sample_index, masked_image_path))
            else:
                print("sample index {} mirror to annotate {}".format(self.sample_index, color_img_path))

            if self.show_plane:
                try:
                    instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)
                    mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path, color_img_path=color_img_path, mirror_mask=instance_mask)
                    mirror_pcd = o3d.geometry.PointCloud()
                    mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                    mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
                except:
                    print("warning : can not generate mesh plane")
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
            option_list.add_option("back n", "BACK : return n times (e.g. back 3 : give up the recent 3 annotated sample and go back)")
            option_list.add_option("goto n", "GOTO : goto the n th image (e.g. goto 3 : go to the third image")
            option_list.add_option("n", "NEXT : goto next image without annotation")
            option_list.add_option("a", "ADJUST: adjust one sample repeatedly")
            option_list.add_option("exit", "EXIT : save and exit")
            option_list.print_option()

            input_option = input()

            mirror_plane = []

            if not option_list.is_input_key_valid(input_option):
                print("invalid input, please input again :D")
                continue
            
            if input_option == "t":
                if current_pcd_path in self.error_list:
                    self.error_list.remove(current_pcd_path)
                self.correct_list.append(current_pcd_path)
                self.save_progress()
                self.get_progress()
            elif input_option == "w":
                if current_pcd_path in self.correct_list:
                    self.correct_list.remove(current_pcd_path)
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
                try:
                    refer_speed = (time.time()-annotation_start_time)/ (self.sample_index - annotation_start_index)
                    left_h = ((len(self.pcd_path_list) - self.sample_index) * refer_speed) / 3600
                    manul_percentage = (manual_adjust_num /  (self.sample_index - annotation_start_index)) * 100
                    print("Reference annotation speed {:.2f} s/sample; Estimate remaining time {:.1f} h; manual adjust {:.2f}%".format(refer_speed, left_h, manul_percentage))
                except:
                    pass
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
            
            elif input_option == "a":
                instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)
                import pdb;pdb.set_trace()
                mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[1,1,0])
                init_step_size = ((np.max(np.array(pcd.points)[:,0])) - (np.min(np.array(pcd.points)[:,0])))/300
                while 1:
                    min_adjust_option_list = Tool_Option()
                    min_adjust_option_list.add_option("f", "FINISH : update refined_sensorD/ refined_meshD/ img_info and EXIT")
                    min_adjust_option_list.add_option("a", "ADJUST : adjust the plane parameter based on current plane parameter")
                    min_adjust_option_list.add_option("i", "INIT : pick 3 points to initialize the plane (press shift + left click to select a point; press shirt + right click to unselect; for more detail please refer to Open3d instruction)")
                    min_adjust_option_list.print_option()
                    min_input_option = input()

                    if min_input_option not in ["f", "i", "a"]:
                        print("invalid input, please input again :D")
                        continue
                    
                    if min_input_option == "f":
                        one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "mirror_plane"), "{}.json".format(img_name))
                        save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_id)
                        manual_adjust_num += 1
                        self.correct_list.append(current_pcd_path)
                        self.save_progress()
                        self.get_progress()
                        break
                    elif min_input_option == "i":
                        [p1, p2, p3] = get_picked_points(pcd)
                        plane_parameter = get_parameter_from_plane_adjustment(pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)
                        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[1,1,0])
                        o3d.visualization.draw_geometries([pcd, mirror_pcd])

                    elif min_input_option == "a":
                        p1 = np.mean(np.array(mirror_pcd.points), axis=0)
                        p2 = np.array(mirror_pcd.points)[0]
                        p3 = np.array(mirror_pcd.points)[-1]
                        if mirror_plane == []:
                            mirror_plane = get_mirror_init_plane_from_3points(p1, p2, p3)
                        plane_parameter = get_parameter_from_plane_adjustment(pcd, mirror_plane, init_step_size)
                        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[1,1,0])
                        o3d.visualization.draw_geometries([pcd, mirror_pcd])


    def anno_move_only_mask(self):
        """
        Move invalid sample to "only_mask" folder
        """
        raw_image_save_folder = os.path.join(self.data_main_folder, "mirror_color_images")
        error_id_path = os.path.join(self.anno_output_folder, "anno_progress", "error_id.txt")
        if os.path.exists(error_id_path):
            self.error_id = read_txt(error_id_path)
        else:
            self.error_id = []
        for color_img_path in self.color_img_list:
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 

            valid_instance = False

            # If this is an invalid sample; only save the RGB image and instance_mask
            if smaple_name in self.error_id:
                command = "find -L {} -type f | grep {}".format(self.data_main_folder, smaple_name)
                for src_path in os.popen(command).readlines():
                    src_path = src_path.strip()
                    dst_path = src_path.replace("with_mirror", "only_mask")
                    dst_folder = os.path.split(dst_path)[0]
                    os.makedirs(dst_folder, exist_ok=True)
                    print("moving {} to only_mask {}".format(src_path, dst_folder))
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.move(src_path, dst_folder)
                
                if self.is_matterport3d:
                    smaple_name = rreplace(smaple_name,"i","d")
                    command = "find -L {} -type f | grep {}".format(self.data_main_folder, smaple_name)
                    for src_path in os.popen(command).readlines():
                        src_path = src_path.strip()
                        dst_path = src_path.replace("with_mirror", "only_mask")
                        dst_folder = os.path.split(dst_path)[0]
                        os.makedirs(dst_folder, exist_ok=True)
                        print("moving {} to only_mask {}".format(src_path, dst_folder))
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        shutil.move(src_path, dst_folder)






    def anno_update_depth_from_imgInfo(self):
        """
        After plane annotation, update "raw_sensorD/raw_meshD" to "refined_sensorD/refined_meshD"

        Output:
            Refined depth saved to refined_sensorD or refined_meshD (Matterport3d only).
        """
        img_info_save_folder = os.path.join(self.anno_output_folder, "mirror_plane")
        error_id_path = os.path.join(self.anno_output_folder, "anno_progress", "error_id.txt")
        if os.path.exists(error_id_path):
            self.error_id = read_txt(error_id_path)
        else:
            self.error_id = []
        for color_img_path in self.color_img_list:
            smaple_name = os.path.split(color_img_path)[1].split(".")[0] 
            mask_img_path = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version),"{}.png".format(smaple_name))
            mask = cv2.imread(mask_img_path)

            one_info_file_path = os.path.join(img_info_save_folder, "{}.json".format(smaple_name))
            info = read_json(one_info_file_path)
            valid_instance = False
            for one_info in info:
                instance_index = one_info["mask_id"]
                instance_index = ImageColor.getcolor("#{}".format(instance_index), "RGB")
                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                plane_parameter = one_info["plane"]

                # Refine mesh raw depth (only Matterport3d have mesh raw depth)
                if self.is_matterport3d:
                    depth_file_name = "{}.png".format(rreplace(smaple_name,"i","d"))
                    raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD",depth_file_name)
                    refined_sensorD_path = os.path.join(self.data_main_folder, "refined_sensorD_{}".format(self.mask_version),depth_file_name)
                    raw_meshD_path = os.path.join(self.data_main_folder, "raw_meshD",depth_file_name)
                    refined_meshD_path = os.path.join(self.data_main_folder, "refined_meshD_{}".format(self.mask_version),depth_file_name)
                    os.makedirs(os.path.split(refined_meshD_path)[0], exist_ok=True)
                    # If there's refined depth; refine the refiend depth 
                    if os.path.exists(refined_meshD_path):
                        raw_meshD_path = refined_meshD_path
                    cv2.imwrite(refined_meshD_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(raw_meshD_path,cv2.IMREAD_ANYDEPTH),self.f))
                    if os.path.exists(refined_meshD_path) and not self.overwrite:
                        continue
                    print("update depth {}".format(refined_meshD_path))
                else:
                    depth_file_name = "{}.png".format(smaple_name)
                # Refine hole raw depth
                raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD",depth_file_name)
                refined_sensorD_path = os.path.join(self.data_main_folder, "refined_sensorD_{}".format(self.mask_version),depth_file_name)
                os.makedirs(os.path.split(refined_sensorD_path)[0], exist_ok=True)
                # If there's refined depth; refine the refiend depth
                if os.path.exists(refined_sensorD_path):
                    raw_sensorD_path = refined_sensorD_path
                if os.path.exists(refined_sensorD_path) and not self.overwrite:
                    continue
                cv2.imwrite(refined_sensorD_path, refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask, cv2.imread(raw_sensorD_path,cv2.IMREAD_ANYDEPTH),self.f))
                print("update depth {}".format(refined_sensorD_path))

    def save_progress(self):
        """Save annotation progress"""
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "anno_progress")
        error_txt_path = os.path.join(anotation_progress_save_folder, "error_list.txt")
        error_id_path = os.path.join(anotation_progress_save_folder, "error_id.txt")
        correct_txt_path = os.path.join(anotation_progress_save_folder, "correct_list.txt")
        save_txt(error_id_path, set(self.error_id))
        save_txt(error_txt_path, set([item  for item in self.error_list]))
        save_txt(correct_txt_path, set([item  for item in self.correct_list]))

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
            self.error_list = [os.path.join(self.data_main_folder, item)  for item in read_txt(error_txt)]
        else:
            self.error_list = []
        
        self.error_id = []
        for item in self.error_list:
            self.error_id.append(item.split("/")[-1].split("_idx_")[0])
        if os.path.exists(correct_txt):
            self.correct_list = [ item  for item in read_txt(correct_txt) ]
        else:
            self.correct_list = []

        for index, one_path in enumerate(self.pcd_path_list):
            if one_path not in self.correct_list and one_path not in self.error_list:
                self.sample_index = index
                return
        self.sample_index = len(self.pcd_path_list)
        return



    def adjust_one_sample_plane(self, instance_index=None, img_name=None):
        """
        Repeatedly adjust one sample's plane parameter

        Args:
            instance_index : "[B]_[G]_[R]",e.g. (128, 0, 0) --> "128_0_0"
            img_name : color image sample name, e.g. 128
        """
        if len(img_name) == 0 or len(instance_index) == 0:
            print("invalid input instance_index {} img_name {}".format(instance_index, img_name))
            exit()

        import open3d as o3d
        if self.is_matterport3d:
            raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(rreplace(img_name, "i", "d")))
            raw_meshD_path = os.path.join(self.data_main_folder, "raw_meshD","{}.png".format(rreplace(img_name, "i", "d")))
            refined_meshD_path = os.path.join(self.anno_output_folder, "refined_meshD_{}".format(self.mask_version),"{}.png".format(rreplace(img_name, "i", "d")))
            refined_sensorD_path = os.path.join(self.anno_output_folder, "refined_sensorD_{}".format(self.mask_version),"{}.png".format(rreplace(img_name, "i", "d")))
        else:
            raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(img_name))
            refined_sensorD_path = os.path.join(self.anno_output_folder, "refined_sensorD_{}".format(self.mask_version),"{}.png".format(img_name))

        color_img_path = os.path.join(self.data_main_folder, "mirror_color_images","{}.jpg".format(img_name))
        mask_path = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version),"{}.png".format(img_name))
        instance_id = [int(i) for i in instance_index.split("_")]
        instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)

        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        instance_tag = img_name + "_idx_" + instance_index
        pcd_path = os.path.join(pcd_save_folder, "{}.ply".format(instance_tag))

        
        one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "mirror_plane"), "{}.json".format(img_name))
        if os.path.exists(one_plane_para_save_path):
            plane_parameter = read_json(one_plane_para_save_path)[instance_index]["plane"]

        pcd = o3d.io.read_point_cloud(pcd_path)
        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f=self.f, plane_parameter=plane_parameter, mirror_mask=instance_mask, color_img_path=color_img_path, color=[0,0,1])
        o3d.visualization.draw_geometries([pcd, mirror_pcd])
        init_step_size = ((np.max(np.array(pcd.points)[:,0])) - (np.min(np.array(pcd.points)[:,0])))/300

        while 1:

            option_list = Tool_Option()
            option_list.add_option("f", "FINISH : update refined_sensorD/ refined_meshD/ img_info and EXIT")
            option_list.add_option("a", "ADJUST : need to adjust the plane parameter")
            option_list.add_option("i", "INIT : pick 3 points to initialize the plane (press shift + left click to select a point; press shirt + right click to unselect; for more detail please refer to Open3d instruction)")
            option_list.print_option()
            input_option = input()

            print("relevant color image path : {}".format(color_img_path))

            if input_option not in ["f", "i", "a"]:
                print("invalid input, please input again :D")
                continue
            
            if input_option == "f":
                save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_id)
                cv2.imwrite(refined_sensorD_path, refine_depth_with_plane_parameter_mask(plane_parameter, instance_mask, cv2.imread(raw_sensorD_path, cv2.IMREAD_ANYDEPTH),self.f))
                if self.is_matterport3d:
                    cv2.imwrite(refined_meshD_path, refine_depth_with_plane_parameter_mask(plane_parameter, instance_mask, cv2.imread(raw_meshD_path, cv2.IMREAD_ANYDEPTH),self.f))
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
            raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(rreplace(img_name, "i", "d")))
            refined_sensorD_path = os.path.join(self.anno_output_folder, "refined_sensorD_{}".format(self.mask_version),"{}.png".format(rreplace(img_name, "i", "d")))
        else:
            raw_sensorD_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(img_name))
            refined_sensorD_path = os.path.join(self.anno_output_folder, "refined_sensorD_{}".format(self.mask_version),"{}.png".format(img_name))

        color_img_path = os.path.join(self.data_main_folder, "mirror_color_images","{}.jpg".format(img_name))
        mask_path = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version),"{}.png".format(img_name))
        instance_id = [int(i) for i in instance_index.split("_")]
        instance_mask = get_grayscale_instanceMask(cv2.imread(mask_path),instance_id)

        pcd_save_folder = os.path.join(self.anno_output_folder, "anno_pcd")
        instance_tag = img_name + "_idx_" + instance_index
        pcd_path = os.path.join(pcd_save_folder, "{}.ply".format(instance_tag))

        
        one_plane_para_save_path = os.path.join(os.path.join(self.anno_output_folder, "mirror_plane"), "{}.json".format(img_name))
        if os.path.exists(one_plane_para_save_path):
            plane_parameter = read_json(one_plane_para_save_path)[instance_index]["plane"]

        refined_depth_to_clamp = cv2.imread(refined_sensorD_path, cv2.IMREAD_ANYDEPTH)
        h, w = refined_depth_to_clamp.shape

        while 1:

            pcd = get_pcd_from_rgb_depthMap(self.f, refined_depth_to_clamp, color_img_path)
            o3d.visualization.draw_geometries([pcd])

            option_list = Tool_Option()
            option_list.add_option("f", "FINISH : update refined_sensorD/ refined_meshD/ img_info and EXIT")
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
                cv2.imwrite(refined_sensorD_path, refined_depth_to_clamp)
                print("annotation of {} finished !".format(refined_sensorD_path))
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
            data_main_folder : Folder raw, raw_sensorD/ raw_meshD, instance_mask saved folder.
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
        if "mp3d" not in data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.color_img_list = [os.path.join(data_main_folder, "mirror_color_images", i) for i in os.listdir(os.path.join(data_main_folder, "mirror_color_images"))]
        self.color_img_list.sort()
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
            Clamped depth : saved to refined_sensorD or mesh_refined depth under self.data_main_folder
        """
        import open3d as o3d
        raw_image_save_folder = os.path.join(self.data_main_folder, "mirror_color_images")
        img_info_save_folder = os.path.join(self.data_main_folder, "mirror_plane")
        for color_img_path in self.color_img_list:
            img_name = os.path.split(color_img_path)[1].split(".")[0] 
            one_info_file_path = os.path.join(img_info_save_folder, "{}.json".format(img_name))
            info = read_json(one_info_file_path)
            for one_info in info.items():
                instance_index_str = one_info[0].split("_")
                instance_index_tuple = [int(i) for i in instance_index_str]

                # Get mask_img_path, depth_img_path, color_img_path
                mask_img_path = os.path.join(self.data_main_folder, "mirror_instance_mask_{}".format(self.mask_version),"{}.png".format(img_name))
                if self.is_matterport3d: 
                    # matterport3d mesh depth don't need to be clamped 
                    depth_img_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(rreplace(img_name, "i", "d")))
                else:
                    depth_img_path = os.path.join(self.data_main_folder, "raw_sensorD","{}.png".format(img_name))
                depth_file_name = depth_img_path.split("/")[-1]
                color_img_path = os.path.join(self.data_main_folder, "mirror_color_images","{}.jpg".format(img_name))
                
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
                    refined_meshD_path = os.path.join(self.data_main_folder, "refined_meshD_{}".format(self.mask_version), depth_file_name)
                    cv2.imwrite(refined_meshD_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=refined_meshD_path, f=self.f, mirror_border_mask=mirror_border_mask , plane_parameter=one_info[1]["plane"], expand_range = self.expand_range, clamp_dis = self.clamp_dis))
                    print("update depth {}".format(refined_meshD_path))

                # Refine hole raw depth
                refined_sensorD_path = os.path.join(self.data_main_folder, "refined_sensorD_{}".format(self.mask_version), depth_file_name)
                cv2.imwrite(refined_sensorD_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=refined_sensorD_path, f=self.f, mirror_border_mask=mirror_border_mask ,plane_parameter=one_info[1]["plane"], expand_range = self.expand_range, clamp_dis = self.clamp_dis))
                print("update depth {}".format(refined_sensorD_path))

    def update_imgInfo_based_on_depth(self):
        """
        Updata img_info based on refined depth

        Output:
            updated img_info : (1) Matterport3d's img_info are updated based on refined_meshD
                               (2) Other datasets img_info are updated based on refined_sensorD
        """
        for color_img_path in self.color_img_list:
            img_name = os.path.split(color_img_path)[1].split(".")[0]
            if self.is_matterport3d:
                depth_img_path = os.path.join(self.data_main_folder, "refined_meshD_{}".format(self.mask_version),"{}.png".format(rreplace(img_name, "i", "d")))
            else:
                depth_img_path = os.path.join(self.data_main_folder, "refined_sensorD_{}".format(self.mask_version),"{}.png".format(img_name))
            mask_img_path = color_img_path.replace("mirror_color_images", "mirror_instance_mask_{}".format(self.mask_version)).replace("jpg","png")
            img_info_path = color_img_path.replace("mirror_color_images", "mirror_plane").replace(".png",".json")
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

    def set_specific_clamp_parameter(self, specific_clamp_parameter_json):
        para = read_json(specific_clamp_parameter_json)
        for one_color_path in self.color_img_list:
            for item in para.items():
                if one_color_path.find(item[0]) > 0:
                    self.expand_range = int(item[1])
                    print("expand range set to {}".format(self.expand_range))
                    return



    def clean_up_repo(self):
        """
        clean_up current folder based on raw image
        """
        raw_folder = os.path.join(self.data_main_folder, "mirror_color_images")
        raw_id_list = [i.split(".")[0] for i in os.listdir(raw_folder)]
        command = "find -L {} -type f ".format(self.data_main_folder)
        for src_path in os.popen(command).readlines():
            is_related_to_color = False
            for one_raw_id in raw_id_list:
                if src_path.find(one_raw_id) > 0:
                    is_related_to_color = True
                    break
                if self.is_matterport3d:
                    one_depth_id = rreplace(one_raw_id)
                    if src_path.find(one_depth_id) > 0:
                        is_related_to_color = True
                        break
            if not is_related_to_color:
                src_path = src_path.strip()
                dst_path = src_path.replace("with_mirror", "clean_up")
                dst_folder = os.path.split(dst_path)[0]
                os.makedirs(dst_folder, exist_ok=True)
                print("moving {} to clean_up {}".format(src_path, dst_folder))
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.move(src_path, dst_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="4")
    parser.add_argument(
        '--data_main_folder', default="")
    parser.add_argument(
        '--dataset', default="mp3d", help="scannet / nyu/ mp3d")
    parser.add_argument(
        '--process_index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    parser.add_argument('--overwrite', help='STEP 1 overwrite current environment setup or not',action='store_true')
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
        '--color_img_list', default="", help="If you want to run anno_move_only_mask() please input a color_img_list")
    parser.add_argument(
        '--specific_clamp_parameter_json', default="", help="json file that contain the sample id and relevant expand_range; only useful for multi-processing")
    parser.add_argument(
        '--instance_index', default="")
    parser.add_argument(
        '--mask_version', default="precise", help="2 mask version : precise/ coarse")
    args = parser.parse_args()

    if args.stage == "1":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.set_overwrite(args.overwrite)
        plane_anno_tool.anno_env_setup()
    elif args.stage == "2":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.set_show_plane(args.anno_show_plane)
        plane_anno_tool.set_overwrite(args.overwrite)
        plane_anno_tool.anno_plane_update_imgInfo()
    elif args.stage == "3":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.set_overwrite(args.overwrite)
        plane_anno_tool.anno_update_depth_from_imgInfo()
    elif args.stage == "4": 
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder, expand_range=args.expand_range, clamp_dis=args.clamp_dis)
        if os.path.exists(args.specific_clamp_parameter_json):
            plane_anno_tool.set_specific_clamp_parameter(args.specific_clamp_parameter_json)
        plane_anno_tool.data_clamping()
    elif args.stage == "all":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.anno_env_setup()
        plane_anno_tool.anno_plane_update_imgInfo()
        assert os.path.exists(args.color_img_list), "please input a valid --color_img_list before moving data based on annotation result"
        plane_anno_tool.set_color_list(args.color_img_list)
        plane_anno_tool.anno_move_only_mask()
        plane_anno_tool.anno_update_depth_from_imgInfo()
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder, expand_range=args.expand_range, clamp_dis=args.clamp_dis)
        if os.path.exists(args.specific_clamp_parameter):
            plane_anno_tool.set_specific_clamp_parameter(args.specific_clamp_parameter)
        plane_anno_tool.data_clamping()
    elif args.stage == "5":
        plane_anno_tool = Data_post_processing(data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.update_imgInfo_based_on_depth()
    elif args.stage == "6":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.adjust_one_sample_plane(args.instance_index, args.img_name)
    elif args.stage == "7":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        plane_anno_tool.manual_clamp_one_sample(args.instance_index, args.img_name)
    elif args.stage == "8":
        plane_anno_tool = Plane_annotation_tool(mask_version=args.mask_version, data_main_folder=args.data_main_folder, process_index=args.process_index, multi_processing=args.multi_processing, border_width=args.border_width, f=args.f, anno_output_folder=args.anno_output_folder)
        assert os.path.exists(args.color_img_list), "please input a valid --color_img_list before moving data based on annotation result"
        plane_anno_tool.set_color_list(args.color_img_list)
        plane_anno_tool.anno_move_only_mask()