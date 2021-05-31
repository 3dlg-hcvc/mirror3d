import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
import sys
from utils.general_utils import *
from utils.algorithm import *
from utils.plane_pcd_utils import *
import json
import shutil
import time
from PIL import ImageColor


class Plane_annotation_tool_new():

    def __init__(self, process_index=0, multi_processing=False, overwrite=True):
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.overwrite = overwrite

    def get_list_to_process(self, full_list):
        full_list.sort()
        if self.multi_processing:
            return full_list[self.process_index:self.process_index+1]
        else:
            return full_list

    def gen_colormask_from_intmask(self, intmask_colormask_txt):
        random.seed(5)
        rand = lambda: random.randint(100, 255)
        BGR_color_list = []
        for i in range(100):
            BGR_color_list.append([rand(), rand(), rand()])
        process_list =  self.get_list_to_process(read_txt(intmask_colormask_txt))
        for item in process_list:
            if len(item.strip().split()) == 2:
                intmask_path, colormask_output_path = item.strip().split()
                os.makedirs(os.path.split(colormask_output_path)[0], exist_ok=True)
                int_mask = cv2.imread(intmask_path, cv2.IMREAD_ANYDEPTH)
                height, width = int_mask.shape
                color_mask = np.zeros((height, width, 3))
                for id in np.unique(int_mask):
                    if id == 0:
                        continue # background
                    color_mask[np.where(int_mask==id)] = BGR_color_list[id-1] # instance id in int_mask start from 1
                cv2.imwrite(colormask_output_path, color_mask)


    def gen_intmask_colormask(self, coco_json, filename_intmask_colormask_txt, coco_filename_tag="file_name"):
        from pycocotools.coco import COCO
        random.seed(5)
        rand = lambda: random.randint(100, 255)
        BGR_color_list = []
        for i in range(100):
            BGR_color_list.append([rand(), rand(), rand()])
        # Get filename intmask_output_path, colormask_output_path dict()
        filename_intmask_colormask_list = read_txt(filename_intmask_colormask_txt)
        color_outputpaths = dict()
        for item in filename_intmask_colormask_list:
            if len(item.strip().split()) == 3:
                colorname , intmask_output_path, colormask_output_path = item.strip().split()
                color_outputpaths[colorname] = [intmask_output_path, colormask_output_path]

        to_gen_list = [i[coco_filename_tag] for i in read_json(coco_json)["images"]]
        to_gen_list = self.get_list_to_process(to_gen_list)

        coco=COCO(coco_json)
        for index in range(len(coco.imgs)):
            img_id = index + 1 # coco image id start from 1
            annIds = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(annIds)
            img_info = coco.loadImgs(img_id)[0]
            intmask_output_path, colormask_output_path = color_outputpaths[img_info[coco_filename_tag]]
            os.makedirs(os.path.split(intmask_output_path)[0], exist_ok=True)
            os.makedirs(os.path.split(colormask_output_path)[0], exist_ok=True)
            int_mask = np.zeros((img_info['height'],img_info['width']))
            color_mask = np.zeros((img_info['height'],img_info['width'], 3))
            for i, ann in enumerate(anns):
                int_mask = coco.annToMask(ann)
                int_mask += (int_mask * (i+1)) # instance id in int_mask start from 1
                color_mask[np.where(int_mask!=0)] = BGR_color_list[i]
            cv2.imwrite(intmask_output_path, int_mask.astype(np.uint16))
            cv2.imwrite(colormask_output_path, color_mask)

    def update_planeinfo_from_depth(self, mask_depth_jsonpath_txt):
        process_list =  self.get_list_to_process(read_txt(mask_depth_jsonpath_txt))
        for item in process_list:
            if len(item.strip().split()) == 4:
                mask_path, depth_path, json_save_path, f = item.strip().split()
                f = int(f)
                mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
                img_info = []
                for instance_index in np.unique(mask):
                    if instance_index == 0:
                        continue
                    binary_instance_mask = (mask == instance_index)
                    mirror_points = (get_points_in_mask(f, depth_path, mirror_mask=binary_instance_mask))
                    plane_parameter = get_mirror_parameter_from_xyzs_by_ransac(mirror_points)
                    one_info = dict()
                    one_info["plane"] = list(plane_parameter)
                    one_info["normal"] = list(unit_vector(list(plane_parameter[:-1])))
                    one_info["mask_id"] = int(instance_index)
                    img_info.append(one_info)
                save_json(json_save_path,img_info)

    def anno_env_setup(self, color_depth_mask_output_f_txt, border_width=25):
        """
        Generate pcd for annotation and initlize plane parameter using ransac
        
        Output:
            pointclouds : .ply file (per instance).
            mirror plane information : .json file (per image); save mirror instances' parameter. 
            color image with a mirror border mask : .png file (per instance).
        """
        def gen_pcd(color_img_path, depth_img_path, mask_img_path, pcd_output_folder, plane_parameter_output_folder, mirror_border_vis_output_folder, f):
            os.makedirs(os.path.split(pcd_output_folder)[0], exist_ok=True)
            os.makedirs(plane_parameter_output_folder, exist_ok=True)
            os.makedirs(mirror_border_vis_output_folder, exist_ok=True)
            int_mask = cv2.imread(intmask_path, cv2.IMREAD_ANYDEPTH)
            for instance_index in np.unique(int_mask):
                if instance_index == 0: # background
                    continue
                pcd_save_name = os.path.split(color_img_path)[-1].split(".")[0] + "_idx_" + str(instance_index)
                pcd_save_path = os.path.join(pcd_output_folder,  "{}.ply".format(pcd_save_name))

                if os.path.isfile(pcd_save_path) and not self.overwrite:
                    print(pcd_save_path , "exist! continue")
                    continue
                else:
                    if os.path.exists(pcd_save_path):
                        print("begin to overwrite {}".format(pcd_save_path))
                    else:
                        print("generating pcd {}".format(pcd_save_path))

                binary_instance_mask = (int_mask == instance_index)
                mirror_border_mask = cv2.dilate(binary_instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width,border_width))) - binary_instance_mask

                #  Save image with masked mirror boreder 
                border_mask_vis_image = visulize_mask_one_image(color_img_path, mirror_border_mask)
                border_mask_vis_output_path = os.path.join(mirror_border_vis_output_folder, "{}.jpg".format(instance_tag)) 
                plt.imsave(border_mask_vis_output_path, border_mask_vis_image)
                print("border_mask_vis_output_path : ", os.path.abspath(border_mask_vis_output_path))

                #  Get pcd with refined mirror depth by ransac 
                pcd, plane_parameter = refine_pcd_by_mirror_border(binary_instance_mask, mirror_border_mask, depth_img_path, color_img_path, f)
                plane_parameter_output_path = os.path.join(plane_parameter_output_folder, "{}.json".format(smaple_name))
                update_plane_parameter_json(plane_parameter, plane_parameter_output_path, instance_index)
                print("plane_parameter saved to :", os.path.abspath(plane_parameter_output_path))

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))

        import open3d as o3d
        process_list =  self.get_list_to_process(read_txt(color_depth_mask_output_f_txt))
        for item in process_list:
            if len(item.strip().split()) == 7:
                color_img_path, depth_img_path, mask_img_path, pcd_output_folder, plane_parameter_output_folder, mirror_border_vis_output_folder, f = item.strip().split()
                f = int(f)
                gen_pcd(color_img_path, depth_img_path, mask_img_path, pcd_output_folder, plane_parameter_output_folder, mirror_border_vis_output_folder, f)

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


    def anno_plane_update_imgInfo(self, anotation_progress_save_folder):
        """
        Check whether mirror plane is correct (verification & adjustment)

        Requirement : open3d 0.10.0 +
        """
        import open3d as o3d
        import warnings
        self.check_file()
        warnings.filterwarnings("ignore")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="3")
    parser.add_argument(
        '--coco_json', default="/project/3dlg-hcvc/mirrors/www/dataset_final_test/nyu/network_input_json/test_10_normal_mirror.json")
    parser.add_argument(
        '--intput_txt', default="/local-scratch/jiaqit/exp/Mirror3D/waste/test_mp3d_improve_stage3.txt")
    parser.add_argument(
        '--height', default=480, type=int, help="image height")
    parser.add_argument(
        '--width', default=640, type=int, help="image width")
    args = parser.parse_args()

    if args.stage == "1":
        print("input txt format: [color image filename in coco json] [integer mask output path] [RGB mask output path]")
        plane_anno_tool = Plane_annotation_tool_new()
        plane_anno_tool.gen_intmask_colormask(args.coco_json, args.intput_txt) 
    if args.stage == "2":
        print("input txt format: [input integer mask path] [RGB mask output path]")
        plane_anno_tool = Plane_annotation_tool_new()
        plane_anno_tool.gen_colormask_from_intmask(args.intput_txt) 
    if args.stage == "3":
        print("input txt format: [input integer mask path] [input refined depth path] [plane JSON file output path] [focal length of this sample]")
        plane_anno_tool = Plane_annotation_tool_new() 
        plane_anno_tool.update_planeinfo_from_depth(args.intput_txt) 
