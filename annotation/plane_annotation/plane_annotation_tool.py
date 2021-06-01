import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import random
import time
import bs4
from skimage import io
from utils.general_utils import *
from utils.algorithm import *
from utils.plane_pcd_utils import *
from PIL import ImageColor


class PlaneAnnotationTool:
    def __init__(self, process_index=0, multi_processing=False, overwrite=True):
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.overwrite = overwrite

    def get_list_to_process(self, full_list):
        full_list.sort()
        if self.multi_processing:
            return full_list[self.process_index:self.process_index + 1]
        else:
            return full_list

    def set_show_plane(self, show_plane):
        """
        For plane annotation: show the mesh plane during annotation or not
        Suggest to show the mesh plane if computer allows
        """
        self.show_plane = show_plane

    def gen_color_mask_from_int_mask(self, int_mask_color_mask_txt):
        random.seed(5)
        rand = lambda: random.randint(100, 255)
        bgr_color_list = []
        for i in range(100):
            bgr_color_list.append([rand(), rand(), rand()])
        process_list = self.get_list_to_process(read_txt(int_mask_color_mask_txt))
        for item in process_list:
            if len(item.strip().split()) == 2:
                int_mask_path, color_mask_output_path = item.strip().split()
                os.makedirs(os.path.split(color_mask_output_path)[0], exist_ok=True)
                int_mask = cv2.imread(int_mask_path, cv2.IMREAD_ANYDEPTH)
                height, width = int_mask.shape
                color_mask = np.zeros((height, width, 3))
                for id in np.unique(int_mask):
                    if id == 0:
                        continue  # background
                    color_mask[np.where(int_mask == id)] = bgr_color_list[
                        id - 1]  # instance id in int_mask start from 1
                cv2.imwrite(color_mask_output_path, color_mask)
                print("RGB instance mask saved to :", color_mask_output_path)

    def gen_int_mask_color_mask(self, coco_json, filename_int_mask_color_mask_txt, coco_filename_tag="file_name"):
        from pycocotools.coco import COCO
        random.seed(5)
        rand = lambda: random.randint(100, 255)
        bgr_color_list = []
        for i in range(100):
            bgr_color_list.append([rand(), rand(), rand()])
        # Get filename int_mask_output_path, color_mask_output_path dict()
        filename_int_mask_color_mask_list = read_txt(filename_int_mask_color_mask_txt)
        color_output_paths = dict()
        for item in filename_int_mask_color_mask_list:
            if len(item.strip().split()) == 3:
                color_name, int_mask_output_path, color_mask_output_path = item.strip().split()
                color_output_paths[color_name] = [int_mask_output_path, color_mask_output_path]

        to_gen_list = [i[coco_filename_tag] for i in read_json(coco_json)["images"]]
        to_gen_list = self.get_list_to_process(to_gen_list)

        coco = COCO(coco_json)
        for index in range(len(coco.imgs)):
            img_id = index + 1  # coco image id start from 1
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            int_mask_output_path, color_mask_output_path = color_output_paths[img_info[coco_filename_tag]]
            os.makedirs(os.path.split(int_mask_output_path)[0], exist_ok=True)
            os.makedirs(os.path.split(color_mask_output_path)[0], exist_ok=True)
            int_mask = np.zeros((img_info['height'], img_info['width']))
            color_mask = np.zeros((img_info['height'], img_info['width'], 3))
            for i, ann in enumerate(anns):
                int_mask = coco.annToMask(ann)
                int_mask += (int_mask * (i + 1))  # instance id in int_mask start from 1
                color_mask[np.where(int_mask != 0)] = bgr_color_list[i]
            cv2.imwrite(int_mask_output_path, int_mask.astype(np.uint16))
            cv2.imwrite(color_mask_output_path, color_mask)

    def update_plane_info_from_depth(self, mask_depth_jsonpath_txt):
        process_list = self.get_list_to_process(read_txt(mask_depth_jsonpath_txt))
        for item in process_list:
            if len(item.strip().split()) == 4:
                mask_path, depth_path, json_save_path, f = item.strip().split()
                f = self.get_and_check_focal_length(f, item)
                mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
                img_info = []
                for instance_index in np.unique(mask):
                    if instance_index == 0:
                        continue
                    binary_instance_mask = (mask == instance_index).astype(np.uint8)
                    mirror_points = (get_points_in_mask(f, depth_path, mirror_mask=binary_instance_mask))
                    plane_parameter = get_mirror_parameter_from_xyzs_by_ransac(mirror_points)
                    one_info = dict()
                    one_info["plane"] = list(plane_parameter)
                    one_info["normal"] = list(unit_vector(list(plane_parameter[:-1])))
                    one_info["mask_id"] = int(instance_index)
                    img_info.append(one_info)
                save_json(json_save_path, img_info)

    def anno_env_setup(self, input_txt, border_width=25):
        """
        Generate pcd for annotation and initlize plane parameter using ransac
        
        Output:
            pointclouds : .ply file (per instance).
            mirror plane information : .json file (per image); save mirror instances' parameter. 
            color image with a mirror border mask : .png file (per instance).
        """

        def gen_pcd(color_img_path, depth_img_path, mask_img_path, pcd_output_folder, plane_parameter_output_path,
                    mirror_border_vis_output_folder, f):
            os.makedirs(mirror_border_vis_output_folder, exist_ok=True)
            os.makedirs(pcd_output_folder, exist_ok=True)
            os.makedirs(os.path.split(plane_parameter_output_path)[0], exist_ok=True)
            int_mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
            for instance_index in np.unique(int_mask):
                if instance_index == 0:  # background
                    continue
                file_save_name = os.path.split(color_img_path)[-1].split(".")[0] + "_idx_" + str(instance_index)
                pcd_save_path = os.path.join(pcd_output_folder, "{}.ply".format(file_save_name))
                if os.path.isfile(pcd_save_path) and not self.overwrite:
                    print(pcd_save_path, "exist! continue")
                    continue
                else:
                    if os.path.exists(pcd_save_path):
                        print("begin to overwrite {}".format(pcd_save_path))
                    else:
                        print("generating pcd {}".format(pcd_save_path))
                binary_instance_mask = (int_mask == instance_index).astype(np.uint8)
                mirror_border_mask = cv2.dilate(binary_instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    border_width, border_width))) - binary_instance_mask

                #  Save image with masked mirror border
                border_mask_vis_image = visulize_mask_one_image(color_img_path, mirror_border_mask)
                border_mask_vis_output_path = os.path.join(mirror_border_vis_output_folder,
                                                           "{}.jpg".format(file_save_name))
                plt.imsave(border_mask_vis_output_path, border_mask_vis_image)
                print("border_mask_vis_output_path : ", os.path.abspath(border_mask_vis_output_path))

                #  Get pcd with refined mirror depth by ransac 
                pcd, plane_parameter = refine_pcd_by_mirror_border(binary_instance_mask, mirror_border_mask,
                                                                   depth_img_path, color_img_path, f)
                update_plane_parameter_json(plane_parameter, plane_parameter_output_path, instance_index)
                print("plane_parameter saved to :", os.path.abspath(plane_parameter_output_path))

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))

        import open3d as o3d
        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) == 7:
                color_img_path, depth_img_path, mask_img_path, pcd_output_folder, \
                plane_parameter_output_path, mirror_border_vis_output_folder, f = item.strip().split()
                f = self.get_and_check_focal_length(f, item)

                if not os.path.exists(color_img_path) or not os.path.exists(depth_img_path) or not os.path.exists(
                        mask_img_path):
                    print("invalid line : ", item)
                    print("input txt format: [input color image path] [input depth image path] [input integer mask "
                          "path] [pointcloud output folder(pointcloud's name will be color image name + instance id)] "
                          "[plane parameter JSON output path] [folder to save color image with mirror border mask] ["
                          "focal length of this sample]")

                gen_pcd(color_img_path, depth_img_path, mask_img_path, pcd_output_folder, plane_parameter_output_path,
                        mirror_border_vis_output_folder, f)

    def save_progress(self, annotation_progress_save_folder):
        """Save annotation progress"""
        error_txt_path = os.path.join(annotation_progress_save_folder, "error_pcd_list.txt")
        correct_txt_path = os.path.join(annotation_progress_save_folder, "correct_pcd_list.txt")
        save_txt(error_txt_path, set([item for item in self.error_pcd_list]))
        save_txt(correct_txt_path, set([item for item in self.correct_pcd_list]))

    def get_progress(self, input_txt, annotation_progress_save_folder):
        """Get annotation progress"""
        self.anno_info_list = []
        self.to_anno_sample_index = 0
        process_list = read_txt(input_txt)
        for item in process_list:
            if len(item.strip().split()) == 7:
                color_img_path, depth_img_path, mask_path, pcd_path, \
                plane_parameter_output_path, mirror_border_vis_path, f = item.strip().split()
                if not os.path.exists(pcd_path) or not os.path.exists(mirror_border_vis_path) or not os.path.exists(
                        color_img_path) or not os.path.exists(depth_img_path) or not os.path.exists(mask_path):
                    print("invalid line : ", item)
                    exit()
                else:
                    self.anno_info_list.append(
                        [color_img_path, depth_img_path, mask_path, pcd_path, plane_parameter_output_path,
                         mirror_border_vis_path, int(f)])

        self.anno_info_list.sort()
        error_txt = os.path.join(annotation_progress_save_folder, "error_pcd_list.txt")
        correct_txt = os.path.join(annotation_progress_save_folder, "correct_pcd_list.txt")

        # get error list
        if os.path.exists(error_txt):
            self.error_pcd_list = read_txt(error_txt)
        else:
            self.error_pcd_list = []

        # get correct list
        if os.path.exists(correct_txt):
            self.correct_pcd_list = read_txt(correct_txt)
        else:
            self.correct_pcd_list = []

        # get error list (regardless of instance id)
        self.error_sample = []
        for item in self.error_pcd_list:
            self.error_sample.append(item.split("_idx_")[0])

        # get annotation start position
        for index, info in enumerate(self.anno_info_list):
            one_path = info[3]  # get pcd path
            if one_path not in self.correct_pcd_list and one_path not in self.error_pcd_list:
                self.to_anno_sample_index = index
                return
        self.to_anno_sample_index = len(self.anno_info_list)
        return

    def anno_plane_update_imgInfo(self, annotation_progress_save_folder, input_txt):
        """
        Plane annotation 

        Requirement : open3d 0.10.0 +
        """
        import open3d as o3d
        import warnings
        warnings.filterwarnings("ignore")
        os.makedirs(annotation_progress_save_folder, exist_ok=True)

        self.get_progress(input_txt, annotation_progress_save_folder)
        annotation_start_index = self.to_anno_sample_index  # self.to_anno_sample_index start from 0
        manual_adjust_num = 0  # count statistic
        annotation_start_time = time.time()
        while 1:
            if self.to_anno_sample_index == len(self.anno_info_list):
                print("annotation finished ! XD")
                exit(1)
            color_img_path, depth_img_path, mask_path, current_pcd_path, \
            plane_parameter_output_path, mirror_border_vis_path, f = self.anno_info_list[self.to_anno_sample_index]
            current_pcd_id = current_pcd_path.split("_idx_")[0]
            mirror_plane = []

            # If one instance in the sample is negative; then this sample is invalid
            if current_pcd_id in self.error_sample:
                self.error_pcd_list.append(current_pcd_path)
                self.save_progress(annotation_progress_save_folder)
                self.get_progress(input_txt, annotation_progress_save_folder)
                print("[AUTO] sample index {} path {} is invalid".format(self.to_anno_sample_index, current_pcd_path))
                continue

            # print the current annotation tag for the sample
            current_sample_status = "N/A"
            if current_pcd_path in self.correct_pcd_list:
                current_sample_status = "correct"
            elif current_pcd_path in self.error_pcd_list:
                current_sample_status = "error"
            print("###################### sample status {} ######################".format(current_sample_status))

            # get the pcd for annotation
            pcd = o3d.io.read_point_cloud(current_pcd_path)
            instance_id = int(current_pcd_path.split("_idx_")[-1].split(".")[0])
            plane_parameter = read_plane_json(plane_parameter_output_path)[instance_id]["plane_parameter"]
            print("sample index {} mirror to annotate {}".format(self.to_anno_sample_index, mirror_border_vis_path))

            # show the point cloud and mesh plane (optional) in the user interface
            if self.show_plane:
                try:
                    instance_mask = (cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH) == instance_id).astype(np.uint8)
                    mirror_points = get_points_in_mask(f=self.f, depth_img_path=depth_img_path,
                                                       color_img_path=color_img_path, mirror_mask=instance_mask)
                    mirror_pcd = o3d.geometry.PointCloud()
                    mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0))
                    mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0)))
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
            option_list.add_option("back n", "BACK : return n times (e.g. back 3 : give up the recent 3 annotated "
                                             "sample and go back)")
            option_list.add_option("goto n", "GOTO : goto the n th image (e.g. goto 3 : go to the third image")
            option_list.add_option("n", "NEXT : goto next image without annotation")
            option_list.add_option("a", "ADJUST: adjust one sample repeatedly")
            option_list.add_option("exit", "EXIT : save and exit")
            option_list.print_option()
            input_option = input()

            if not option_list.is_input_key_valid(input_option):
                print("invalid input, please input again :D")
                continue

            if input_option == "t":
                if current_pcd_path in self.error_pcd_list:
                    self.error_pcd_list.remove(current_pcd_path)
                self.correct_pcd_list.append(current_pcd_path)
                self.save_progress(annotation_progress_save_folder)
                self.get_progress(input_txt, annotation_progress_save_folder)

            elif input_option == "w":
                if current_pcd_path in self.correct_pcd_list:
                    self.correct_pcd_list.remove(current_pcd_path)
                self.error_pcd_list.append(current_pcd_path)
                self.save_progress(annotation_progress_save_folder)
                self.get_progress(input_txt, annotation_progress_save_folder)
            elif input_option == "n":
                if current_sample_status == "N/A":
                    print("please annotate current sample :-)")
                    continue
                self.to_anno_sample_index += 1
            elif input_option == "exit":
                self.save_progress(annotation_progress_save_folder)
                self.get_progress(input_txt, annotation_progress_save_folder)
                print("current progress {} / {}".format(self.to_anno_sample_index, len(self.anno_info_list)))
                refer_speed = (time.time() - annotation_start_time) / (
                        self.to_anno_sample_index - annotation_start_index)
                left_h = ((len(self.anno_info_list) - self.to_anno_sample_index) * refer_speed) / 3600
                manual_percentage = (manual_adjust_num / (self.to_anno_sample_index - annotation_start_index)) * 100
                print("Reference annotation speed {:.2f} s/sample; "
                      "Estimate remaining time {:.1f} h; manual adjust {:.2f}%"
                      .format(refer_speed, left_h, manual_percentage))
                exit(1)
            elif "back" in input_option:
                n = int(input_option.split()[1]) - 1
                if self.to_anno_sample_index - n < 0:
                    print("at most return {} times".format(self.to_anno_sample_index + 1))
                    continue
                self.to_anno_sample_index -= n
            elif "goto" in input_option:
                n = int(input_option.split()[1]) - 1
                if n > len(self.anno_info_list) - 1:
                    print("you can go to 0 ~ {}".format(len(self.anno_info_list) - 1))
                    continue
                self.to_anno_sample_index = n
            elif input_option == "a":
                instance_mask = (cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH) == instance_id).astype(np.uint8)
                mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f, plane_parameter=plane_parameter,
                                                                      mirror_mask=instance_mask,
                                                                      color_img_path=color_img_path, color=[1, 1, 0])
                init_step_size = ((np.max(np.array(pcd.points)[:, 0])) - (np.min(np.array(pcd.points)[:, 0]))) / 300
                while 1:
                    min_adjust_option_list = Tool_Option()
                    min_adjust_option_list.add_option("f", "FINISH : update refined_sensorD/ refined_meshD/ img_info "
                                                           "and EXIT")
                    min_adjust_option_list.add_option("a", "ADJUST : adjust the plane parameter based on current "
                                                           "plane parameter")
                    min_adjust_option_list.add_option("i", "INIT : pick 3 points to initialize the plane (press shift "
                                                           "+ left click to select a point; press shirt + right click "
                                                           "to unselect; for more detail please refer to Open3d "
                                                           "instruction)")
                    min_adjust_option_list.print_option()
                    min_input_option = input()

                    if min_input_option not in ["f", "i", "a"]:
                        print("invalid input, please input again :D")
                        continue

                    if min_input_option == "f":
                        update_plane_parameter_json(plane_parameter, plane_parameter_output_path, instance_id)
                        manual_adjust_num += 1
                        self.correct_pcd_list.append(current_pcd_path)
                        self.save_progress(annotation_progress_save_folder)
                        self.get_progress(input_txt, annotation_progress_save_folder)
                        break
                    elif min_input_option == "i":
                        [p1, p2, p3] = get_picked_points(pcd)
                        plane_parameter = get_parameter_from_plane_adjustment(
                            pcd, get_mirror_init_plane_from_3points(p1, p2, p3), init_step_size)
                        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f, plane_parameter=plane_parameter,
                                                                              mirror_mask=instance_mask,
                                                                              color_img_path=color_img_path,
                                                                              color=[1, 1, 0])
                        o3d.visualization.draw_geometries([pcd, mirror_pcd])

                    elif min_input_option == "a":
                        p1 = np.mean(np.array(mirror_pcd.points), axis=0)
                        p2 = np.array(mirror_pcd.points)[0]
                        p3 = np.array(mirror_pcd.points)[-1]
                        if not mirror_plane:
                            mirror_plane = get_mirror_init_plane_from_3points(p1, p2, p3)
                        plane_parameter = get_parameter_from_plane_adjustment(pcd, mirror_plane, init_step_size)
                        mirror_pcd = get_mirrorPoint_based_on_plane_parameter(f, plane_parameter=plane_parameter,
                                                                              mirror_mask=instance_mask,
                                                                              color_img_path=color_img_path,
                                                                              color=[1, 1, 0])
                        o3d.visualization.draw_geometries([pcd, mirror_pcd])

    def get_and_check_focal_length(self, f, line):
        try:
            f = int(f)
            return f
        except:
            print("{} invalid focal length format".format(f))
            print("please check line: ", line)
            exit()

    def anno_update_depth_from_img_info(self, input_txt):
        """
        After plane annotation, update "raw_sensorD/raw_meshD" to "refined_sensorD/refined_meshD"

        Output:
            Refined depth saved to refined_sensorD or refined_meshD (Matterport3d only).
        """
        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 5:
                continue
            rawD_path, mask_img_path, plane_parameter_json_path, refD_output_path, f = item.strip().split()
            if not os.path.exists(rawD_path) or not os.path.exists(mask_img_path) or not os.path.exists(
                    plane_parameter_json_path):
                print("invalid line : ", item)
                print("input txt format: [path to depth map to refine (rawD)] [input integer mask path] [plane "
                      "parameter JSON output path] [path to save the refined depth map (refD)] [focal length of this "
                      "sample]")
                continue
            f = self.get_and_check_focal_length(f, item)

            os.makedirs(os.path.split(refD_output_path)[0], exist_ok=True)
            mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
            info = read_json(plane_parameter_json_path)
            valid_instance = False
            for one_info in info:
                instance_index = one_info["mask_id"]
                binary_instance_mask = (mask == instance_index).astype(np.uint8)
                plane_parameter = one_info["plane"]
                cv2.imwrite(refD_output_path,
                            refine_depth_with_plane_parameter_mask(plane_parameter, binary_instance_mask,
                                                                   cv2.imread(rawD_path, cv2.IMREAD_ANYDEPTH), f))
                print("update depth {}".format(refD_output_path))

    def data_clamping(self, input_txt, expand_range=100, clamp_dis=100, border_width=25):
        """
        Clamp data based on 3D bbox

        Output:
            Clamped depth : saved to refined_sensorD or mesh_refined depth under self.data_main_folder
        """
        import open3d as o3d
        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 5:
                continue
            refD_path, mask_img_path, plane_parameter_json_path, clamped_refD_path, f = item.strip().split()
            if not os.path.exists(refD_path) or not os.path.exists(mask_img_path) or not os.path.exists(
                    plane_parameter_json_path):
                print("invalid line : ", item)
                print("input txt format: [path to depth map to the unclamped refine (rawD)] [input integer mask path] "
                      "[plane parameter JSON output path] [path to save the clamped refined depth map (refD)] [focal "
                      "length of this sample]")
                continue
            f = self.get_and_check_focal_length(f, item)
            mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
            for instance_index in np.unique(mask):
                if instance_index == 0:
                    continue  # background

                # Get mirror_border_mask
                instance_mask = (mask == instance_index).astype(np.uint8)
                mirror_border_mask = cv2.dilate(instance_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    border_width, border_width))) - cv2.erode(instance_mask,
                                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

                # Get mirror_bbox
                mirror_points = get_points_in_mask(f=f, depth_img_path=refD_path, mirror_mask=instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0)))

                # Get plane parameter
                plane_parameter = read_plane_json(plane_parameter_json_path)[instance_index]["plane_parameter"]

                # Refine hole raw depth
                os.makedirs(os.path.split(clamped_refD_path)[0], exist_ok=True)
                cv2.imwrite(clamped_refD_path, clamp_pcd_by_bbox(mirror_bbox=mirror_bbox, depth_img_path=refD_path, f=f,
                                                                 mirror_border_mask=mirror_border_mask,
                                                                 plane_parameter=plane_parameter,
                                                                 expand_range=expand_range, clamp_dis=clamp_dis))
                print("update depth {}".format(clamped_refD_path))

    def generate_pcdMesh_for_vis(self, input_txt):
        """
        Generate "point cloud" + "mesh plane" for specific sample

        Output:
            "point cloud" + "mesh plane" : Saved under self.output_folder.
        """

        import open3d as o3d
        # Pack as a function to better support Matterport3d ply generation
        def generate_and_save_ply(color_img_path, depth_img_path, mask_img_path, plane_parameter_json_path,
                                  pcd_save_folder, mesh_save_folder, f):
            os.makedirs(pcd_save_folder, exist_ok=True)
            os.makedirs(mesh_save_folder, exist_ok=True)

            mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
            #  Get pcd and masked RGB image for each instance
            for instance_index in np.unique(mask):
                if instance_index == 0:  # background
                    continue
                save_name = color_img_path.split("/")[-1].split(".")[0] + "_idx_" + str(instance_index)
                mesh_save_path = os.path.join(mesh_save_folder, "{}.ply".format(save_name))
                pcd_save_path = os.path.join(pcd_save_folder, "{}.ply".format(save_name))
                binary_instance_mask = (mask == instance_index).astype(np.uint8)
                plane_parameter = read_plane_json(plane_parameter_json_path)[instance_index]["plane_parameter"]

                if os.path.exists(pcd_save_path) and os.path.exists(mesh_save_path) and not self.overwrite:
                    print(pcd_save_path, mesh_save_path, "exist! continue")
                    return

                # Get pcd for the instance
                pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)

                # Get mirror plane for the instance
                mirror_points = get_points_in_mask(f, depth_img_path, mirror_mask=binary_instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0)))
                mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox)

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))

                o3d.io.write_triangle_mesh(mesh_save_path, mirror_plane)
                print("mirror plane (mesh) saved  to :", os.path.abspath(mesh_save_path))

        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 7:
                continue
            color_img_path, depth_img_path, mask_img_path, plane_parameter_json_path, \
            pcd_save_folder, mesh_save_folder, f = item.strip().split()
            if not os.path.exists(color_img_path) or not os.path.exists(depth_img_path) or not os.path.exists(
                    plane_parameter_json_path):
                print("invalid line : ", item)
                print("input txt format: [input color image path] [input depth image path] [input integer mask path] "
                      "[plane parameter JSON path] [folder to save the output pointcloud] [folder to save the output "
                      "mesh plane] [focal length of this sample]")
                continue
            f = self.get_and_check_focal_length(f, item)
            generate_and_save_ply(color_img_path, depth_img_path, mask_img_path, plane_parameter_json_path,
                                  pcd_save_folder, mesh_save_folder, f)

    def set_view_mode(self, view_mode):
        """Function to save the view mode"""
        self.view_mode = view_mode

    @staticmethod
    def rotate_pcd_mesh_topdown(screenshot_output_folder, pcd, plane, above_height=3000):
        """
        Rotate the "pcd + mesh" by topdown view

        Output:
            Screenshots png
        """
        import open3d as o3d

        screenshot_id = 0
        mesh_center = np.mean(np.array(plane.vertices), axis=0)
        rotation_step_degree = 10
        start_rotation = get_extrinsic(90, 0, 0, [0, 0, 0])
        step_translation = get_extrinsic(0, 0, 0, [-mesh_center[0], -mesh_center[1] + above_height, -mesh_center[2]])
        start_position = np.dot(start_rotation, step_translation)

        def rotate_view(vis):
            nonlocal screenshot_id
            t_rotate = get_extrinsic(0, rotation_step_degree * (screenshot_id + 1), 0, [0, 0, 0])
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.dot(np.dot(start_rotation, t_rotate), step_translation)
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            screenshot_id += 1
            screenshot_save_path = os.path.join(screenshot_output_folder, "{0:05d}.png".format(screenshot_id))
            vis.capture_screen_image(filename=screenshot_save_path, do_render=True)
            print("image saved to {}".format(screenshot_save_path))
            if screenshot_id > (360 / rotation_step_degree):
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view)
        vis.create_window(width=800, height=800)
        vis.get_render_option().point_size = 1.0
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = start_position
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        vis.run()

    @staticmethod
    def rotate_pcd_mesh_front(screenshot_output_folder, pcd, plane):
        """
        Rotate the "pcd + mesh" by front view

        Output:
            Screenshots png
        """
        import open3d as o3d

        screenshot_id = 0
        mesh_center = np.mean(np.array(plane.vertices), axis=0)
        rotation_step_degree = 10
        start_position = get_extrinsic(0, 0, 0, [0, 0, 3000])

        def rotate_view(vis):
            nonlocal screenshot_id
            t_to_center = get_extrinsic(0, 0, 0, mesh_center)
            t_rotate = get_extrinsic(0, rotation_step_degree * (screenshot_id + 1), 0, [0, 0, 0])
            t_to_mesh = get_extrinsic(0, 0, 0, -mesh_center)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.dot(start_position, np.dot(np.dot(t_to_center, t_rotate), t_to_mesh))
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            screenshot_id += 1
            screenshot_save_path = os.path.join(screenshot_output_folder, "{0:05d}.png".format(screenshot_id))
            vis.capture_screen_image(filename=screenshot_save_path, do_render=True)
            print("image saved to {}".format(screenshot_save_path))
            if screenshot_id > (360 / rotation_step_degree):
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view)
        vis.create_window(width=800, height=800)
        vis.get_render_option().point_size = 1.0
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = start_position
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        vis.run()

    def generate_video_screenshot_from_pcdMesh(self, input_txt, above_height=3000):
        """
        Generate "pcd + mesh"'s screenshots

        Args:
            self.view_mode : str; "topdown" / "front".

        Output:
            screenshots png
        """
        import open3d as o3d

        def generate_video_ffmpeg(one_video_save_path, one_screenshot_output_folder):
            os.makedirs(os.path.split(one_video_save_path)[0], exist_ok=True)
            try:
                start_time = time.time()
                if os.path.exists(one_video_save_path):
                    if not self.overwrite:
                        print("{} video exists!".format(one_video_save_path))
                        return
                    else:
                        os.remove(one_video_save_path)
                command = "ffmpeg -f image2 -i " + one_screenshot_output_folder + "/%05d.png " + one_video_save_path
                os.system(command)
                print("video saved to {}, used time :{}".format(one_video_save_path, time.time() - start_time))
            except:
                print("error saving video for :", one_screenshot_output_folder)

        def generate_screenshot(pcd_path, mesh_path, screenshot_output_folder):

            pcd = o3d.io.read_point_cloud(pcd_path)
            mirror_plane = o3d.io.read_triangle_mesh(mesh_path)
            os.makedirs(screenshot_output_folder, exist_ok=True)

            if len(os.listdir(screenshot_output_folder)) == 37 and not self.overwrite:
                print("screenshots for {} exist ! continue".format(pcd_path))
                return

            if self.view_mode == "topdown":
                topdown_folder = os.path.join(screenshot_output_folder, "topdown")
                os.makedirs(topdown_folder, exist_ok=True)
                self.rotate_pcd_mesh_topdown(topdown_folder, pcd, mirror_plane, above_height)
            else:
                front_folder = os.path.join(screenshot_output_folder, "front")
                os.makedirs(front_folder, exist_ok=True)
                self.rotate_pcd_mesh_front(front_folder, pcd, mirror_plane)

        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 3:
                continue
            pcd_path, mesh_path, screenshot_output_folder = item.strip().split()
            if not os.path.exists(pcd_path) or not os.path.exists(mesh_path):
                print("invalid line : ", item)
                print("input txt format: [input color image path] [input depth image path] [input integer mask path] "
                      "[plane parameter JSON path] [folder to save the output pointcloud] [folder to save the output "
                      "mesh plane] [focal length of this sample]")
                continue
            generate_screenshot(pcd_path, mesh_path, screenshot_output_folder)
            if self.view_mode == "topdown":
                topdown_folder = os.path.join(screenshot_output_folder, "topdown")
                video_save_path = os.path.join(screenshot_output_folder,
                                               "topdown_{}_.mp4".format(os.path.split(mesh_path)[-1].split(".")[0]))
                generate_video_ffmpeg(video_save_path, topdown_folder)
            else:
                front_folder = os.path.join(screenshot_output_folder, "front")
                video_save_path = os.path.join(screenshot_output_folder,
                                               "front_{}_.mp4".format(os.path.split(mesh_path)[-1].split(".")[0]))
                generate_video_ffmpeg(video_save_path, front_folder)

    def gen_verification_html(self, input_txt, video_num_per_page, html_output_folder):

        template_path = "visualization/template/veri_template.html"
        os.makedirs(html_output_folder, exist_ok=True)
        process_list_temp = self.get_list_to_process(read_txt(input_txt))
        process_list = []
        for item in process_list_temp:
            if len(item.strip().split()) != 5:
                continue
            sample_id, color_img_path, colored_depth_path, front_video_path, topdown_video_path = item.strip().split()
            if not os.path.exists(color_img_path) or not os.path.exists(colored_depth_path) or not os.path.exists(
                    front_video_path) or not os.path.exists(topdown_video_path):
                print("invalid line : ", item)
                print("input txt format: [sample id] [input color image path] [colored depth map saved path] [front "
                      "view video path] [topdown view video path]")
                continue

            process_list.append(item.strip().split())
        process_sub_list = [process_list[x:x + video_num_per_page] for x in
                            range(0, len(process_list), video_num_per_page)]
        for html_index, process_sub in enumerate(process_sub_list):

            with open(template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")

            new_table = soup.new_tag("table")
            new_table["style"] = "width: 100%%; margin-left: auto; margin-right: auto;"
            soup.body.div.append(new_table)

            # add heading 
            heading_tag = ["ID", "Color Image", "Depth Image", "Topdown View Pointcloud", "Front View Pointcloud"]
            heading = soup.new_tag("tr")

            for item_index, tag in enumerate(heading_tag):
                heading["class"] = "one-item"
                one_heading = soup.new_tag("td")
                text = soup.new_tag("p")
                text.string = tag
                text["style"] = "text-align: center;"
                one_heading.append(text)
                heading.append(one_heading)
            new_table.append(heading)
            for one_sub_info in process_sub:
                sample_id, color_img_path, colored_depth_path, front_video_path, topdown_video_path = one_sub_info

                # append sample_id
                new_tr = soup.new_tag("tr")
                sample_id_box = soup.new_tag("td")
                text = soup.new_tag("p")
                text.string = sample_id
                text["style"] = "text-align: center; font-size: 50px;"
                sample_id_box.append(text)
                new_tr.append(sample_id_box)

                # append color image to one line in HTML
                one_color_img = soup.new_tag("td")
                one_color_img["class"] = "one-item"
                img = soup.new_tag('img', src=os.path.relpath(color_img_path, html_output_folder))
                img["style"] = "max-height: 220px; width:100%;"
                one_color_img.append(img)
                new_tr.append(one_color_img)

                # append colored depth image to one line in HTML
                one_color_img = soup.new_tag("td")
                one_color_img["class"] = "one-item"
                img = soup.new_tag('img', src=os.path.relpath(colored_depth_path, html_output_folder))
                img["style"] = "max-height: 220px; width:100%;"
                one_color_img.append(img)
                new_tr.append(one_color_img)

                # add topdown video
                video_td = soup.new_tag("td")
                video_td["class"] = "one-item"
                one_video = soup.new_tag("video")
                one_video["class"] = "lazy-video"
                one_video["controls"] = "True"
                one_video["autoplay"] = "True"
                one_video["muted"] = "True"
                one_video["loop"] = "True"
                new_link = soup.new_tag("source")
                new_link["data-src"] = os.path.relpath(topdown_video_path, html_output_folder)
                new_link["type"] = "video/mp4"
                one_video.append(new_link)
                video_td.append(one_video)
                new_tr.append(video_td)

                # add front video
                video_td = soup.new_tag("td")
                video_td["class"] = "one-item"
                one_video = soup.new_tag("video")
                one_video["class"] = "lazy-video"
                one_video["controls"] = "True"
                one_video["autoplay"] = "True"
                one_video["muted"] = "True"
                one_video["loop"] = "True"
                new_link = soup.new_tag("source")
                new_link["data-src"] = os.path.relpath(front_video_path, html_output_folder)
                new_link["type"] = "video/mp4"
                one_video.append(new_link)
                video_td.append(one_video)
                new_tr.append(video_td)
                new_table.append(new_tr)

            html_path = os.path.join(html_output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)
            print("html saved to :", os.path.abspath(html_path))
            print("debug : ", html_path.replace("/project/3dlg-hcvc/mirrors/www",
                                                "https://aspis.cmpt.sfu.ca/projects/mirrors"))

    def gen_colored_grayscale_for_depth(self, input_txt):
        """
        Generate colored depth for one sample
        Output:
            colored depth image (using plt "magma" colormap)
        """

        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 2:
                continue
            depth_path, colored_depth_output_path = item.strip().split()
            if not os.path.exists(depth_path):
                print("invalid line : ", item)
                print("input txt format: [input depth image path] [colored depth map saved path]")
                continue
            os.makedirs(os.path.split(colored_depth_output_path)[0], exist_ok=True)
            save_heatmap_no_border(cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH), colored_depth_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--function', default="3")
    parser.add_argument(
        '--coco_json', default="")
    parser.add_argument(
        '--annotation_progress_save_folder', default="",
        help="folder to save the plane annotation progress")
    parser.add_argument(
        '--input_txt', default="")
    parser.add_argument('--multi_processing', help='do multi-process or not', action='store_true')
    parser.add_argument('--overwrite', help='overwrite current result or not', action='store_true')
    parser.add_argument('--anno_show_plane', help='do multi-process or not', action='store_true')
    parser.add_argument(
        '--process_index', default=0, type=int, help="if do --multi_processing please input the process index")
    parser.add_argument(
        '--border_width', default=25, type=int,
        help="border width of mirror; when setup annotation environment, specify a border with to run RANSAC on "
             "mirror border")
    parser.add_argument(
        '--expand_range', default=200, type=int, help="expand the mirror instance bbox by expand_range; unit : mm")
    parser.add_argument(
        '--clamp_dis', default=100, type=int, help="outliers threshold")
    parser.add_argument(
        '--above_height', default=3000, type=int, help="camera height to the mirror plane center in the topdown view")
    parser.add_argument(
        '--video_num_per_page', default=100, type=int)
    parser.add_argument(
        '--html_output_folder', default="")
    parser.add_argument(
        '--view_mode', default="front", help="object view angle : (1) topdown (2) front")
    args = parser.parse_args()

    plane_anno_tool = PlaneAnnotationTool(process_index=args.process_index, multi_processing=args.multi_processing,
                                          overwrite=args.overwrite)

    if args.function == "1":
        print("input txt format: [color image filename in coco json] [integer mask output path] [RGB mask output path]")
        plane_anno_tool.gen_int_mask_color_mask(args.coco_json, args.input_txt)
    elif args.function == "2":
        print("input txt format: [input integer mask path] [RGB mask output path]")
        plane_anno_tool.gen_color_mask_from_int_mask(args.input_txt)
    elif args.function == "3":
        print("input txt format: [input integer mask path] [input refined depth path] [plane JSON file output path] ["
              "focal length of this sample]")
        plane_anno_tool.update_plane_info_from_depth(args.input_txt)
    elif args.function == "4":
        print("input txt format: [input color image path] [input depth image path] [input integer mask path] ["
              "pointcloud output folder(pointcloud's name will be color image name + instance id)] [plane parameter "
              "JSON output path] [folder to save color image with mirror border mask] [focal length of this sample]")
        plane_anno_tool.anno_env_setup(args.input_txt, args.border_width)
    elif args.function == "5":
        print("input txt format: [input color image path] [input depth image path] [input integer mask path] ["
              "instance pointcloud path] [plane parameter JSON output path] [path to the color image with mirror "
              "border mask] [focal length of this sample]")
        plane_anno_tool.set_show_plane(args.anno_show_plane)
        plane_anno_tool.anno_plane_update_imgInfo(args.annotation_progress_save_folder, args.input_txt)
    elif args.function == "6":
        print("input txt format: [path to depth map to refine (rawD)] [input integer mask path] [plane parameter JSON "
              "output path] [path to save the refined depth map (refD)] [focal length of this sample]")
        plane_anno_tool.anno_update_depth_from_img_info(args.input_txt)
    elif args.function == "7":
        print("input txt format: [path to depth map to the unclamped refine (rawD)] [input integer mask path] [plane "
              "parameter JSON output path] [path to save the clamped refined depth map (refD)] [focal length of this "
              "sample]")
        plane_anno_tool.data_clamping(args.input_txt, args.expand_range, args.clamp_dis, args.border_width)
    elif args.function == "8":
        print("input txt format: [input color image path] [input depth image path] [input integer mask path] [plane "
              "parameter JSON path] [folder to save the output pointcloud] [folder to save the output mesh plane] ["
              "focal length of this sample]")
        plane_anno_tool.generate_pcdMesh_for_vis(args.input_txt)
    elif args.function == "9":
        print("input txt format: [path to pointcloud] [path to mesh plane] [screenshot output main folder]")
        plane_anno_tool.set_view_mode("topdown")
        plane_anno_tool.generate_video_screenshot_from_pcdMesh(args.input_txt, args.above_height)
        plane_anno_tool.set_view_mode("front")
        plane_anno_tool.generate_video_screenshot_from_pcdMesh(args.input_txt, args.above_height)
    elif args.function == "10":
        print("input txt format: [path to pointcloud] [path to mesh plane] [screenshot output main folder]")
        plane_anno_tool.set_view_mode(args.view_mode)
        plane_anno_tool.generate_video_screenshot_from_pcdMesh(args.input_txt, args.above_height)
    elif args.function == "11":
        print("input txt format: [input depth image path] [colored depth map saved path]")
        plane_anno_tool.gen_colored_grayscale_for_depth(args.input_txt)
    elif args.function == "12":
        print("input txt format: [sample id] [input color image path] [colored depth map saved path] [front view "
              "video path] [topdown view video path]")
        plane_anno_tool.gen_verification_html(args.input_txt, args.video_num_per_page, args.html_output_folder)
