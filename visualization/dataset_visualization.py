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
from annotation.plane_annotation_tool.plane_annotation_tool import *
 

class Dataset_visulization(Plane_annotation_tool):

    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, 
                f=519, output_folder=None, overwrite=True, window_w=800, window_h=800, view_mode="topdown"):
        """
        Initilization

        Args:
            data_main_folder : Folder raw, hole_raw_depth/ mesh_raw_depth, instance_mask saved folder.
            output_folder(optional) : Inital pcd, img_info, border_vis saved forder (default : data_main_folder).
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
        self.overwrite = overwrite
        self.window_w = window_w
        self.window_h = window_h
        self.view_mode = view_mode
        
        if "m3d" not in self.data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.color_img_list = [os.path.join(self.data_main_folder, "raw", i) for i in os.listdir(os.path.join(self.data_main_folder, "raw"))]
        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.f = f
        if output_folder == None or not os.path.exists(output_folder):
            self.output_folder = self.data_main_folder
            print("########## NOTE output saved to {}, this may overwrite your current information ############".format(self.output_folder))
        else:
            self.output_folder = output_folder
        self.error_info_path = os.path.join(self.output_folder, "error_img_list.txt")

    def generate_colored_depth_for_whole_dataset(self):
        """
        Call function self.generate_colored_depth_for_one_GTsample 
            to colored depth image for all sample under self.data_main_folders
        """
        for one_color_img_path in self.color_img_list:
            self.generate_colored_depth_for_one_GTsample(one_color_img_path)
    

    def generate_colored_depth_for_one_GTsample(self, color_img_path):
        """
        Generate colored depth for one sample
        Output:
            colored depth image (using plt "magma" colormap)
        """
        if self.is_matterport3d:
            sample_name = rreplace(color_img_path.split("/")[-1], "i", "d")
            colored_depth_save_folder = os.path.join(self.output_folder, "hole_refined_colored_depth")
            os.makedirs(colored_depth_save_folder, exist_ok=True)
            ori_depth = os.path.join(self.data_main_folder, "hole_refined_depth", sample_name)
            colored_depth_save_path = os.path.join(colored_depth_save_folder, sample_name)
            save_heatmap_no_border(cv2.imread(ori_depth, cv2.IMREAD_ANYDEPTH), colored_depth_save_path)

            colored_depth_save_folder = os.path.join(self.output_folder, "mesh_refined_colored_depth")
            os.makedirs(colored_depth_save_folder, exist_ok=True)
            ori_depth = os.path.join(self.data_main_folder, "mesh_refined_depth", sample_name)
            colored_depth_save_path = os.path.join(colored_depth_save_folder, sample_name)
            save_heatmap_no_border(cv2.imread(ori_depth, cv2.IMREAD_ANYDEPTH), colored_depth_save_path)
        else:
            sample_name = color_img_path.split("/")[-1]
            colored_depth_save_folder = os.path.join(self.output_folder, "hole_refined_colored_depth")
            os.makedirs(colored_depth_save_folder, exist_ok=True)
            ori_depth = os.path.join(self.data_main_folder, "hole_refined_depth", sample_name)
            colored_depth_save_path = os.path.join(colored_depth_save_folder, sample_name)
            save_heatmap_no_border(cv2.imread(ori_depth, cv2.IMREAD_ANYDEPTH), colored_depth_save_path)




    def generate_pcdMesh_for_whole_dataset(self):
        """
        Call function self.generate_pcdMesh_for_one_GTsample 
            to generate mesh.ply and pcd.ply for all sample under self.data_main_folders
        """
        for one_color_img_path in self.color_img_list:
            self.generate_pcdMesh_for_one_GTsample(one_color_img_path)


    def generate_pcdMesh_for_one_GTsample(self, color_img_path):
        """
        Generate "point cloud" + "mesh plane" for specific sample

        Args:
            color_img_path : The color image absolute path for the specific sample.
        
        Output:
            "point cloud" + "mesh plane" : Saved under self.output_folder.
        """

        import open3d as o3d
        # Pack as a function to better support Matterport3d ply generation
        def generate_and_save_ply(depth_img_path, ply_save_folder):
            pcd_save_folder = os.path.join(ply_save_folder, "pcd")
            mesh_save_folder = os.path.join(ply_save_folder, "mesh")
            os.makedirs(pcd_save_folder, exist_ok=True)
            os.makedirs(mesh_save_folder, exist_ok=True)

            mask_img_path = color_img_path.replace("raw","instance_mask")
            mask = cv2.imread(mask_img_path)
            img_info_path = color_img_path.replace("raw","img_info").replace("png","json")
            one_img_info = read_json(img_info_path)
            
            #  Get pcd and masked RGB image for each instance
            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue

                instance_tag = "_idx"
                for i in instance_index:
                    instance_tag += "_{}".format(i)
                instance_tag = color_img_path.split("/")[-1].split(".")[0] + instance_tag
                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                plane_parameter = one_img_info[instance_tag.split("_idx_")[1]]

                # Get pcd for the instance
                pcd = get_pcd_from_rgbd_depthPath(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)

                # Get mirror plane for the instance
                mirror_points = get_points_in_mask(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
                mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter["plane_parameter"], mirror_bbox)

                pcd_save_path = os.path.join(pcd_save_folder,  "{}.ply".format(instance_tag))
                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))

                mesh_save_path = os.path.join(mesh_save_folder,  "{}.ply".format(instance_tag))
                o3d.io.write_triangle_mesh(mesh_save_path, mirror_plane)
                print("mirror plane (mesh) saved  to :", os.path.abspath(mesh_save_path))

        if self.is_matterport3d:
                depth_img_path = rreplace(color_img_path.replace("raw","hole_refined_depth"), "i", "d")
                ply_save_folder = os.path.join(self.output_folder, "hole_refined_ply")
                os.makedirs(ply_save_folder, exist_ok=True)
                generate_and_save_ply(depth_img_path, ply_save_folder)

                depth_img_path = rreplace(color_img_path.replace("raw","mesh_refined_depth"), "i", "d")
                ply_save_folder = os.path.join(self.output_folder, "mesh_refined_ply")
                os.makedirs(ply_save_folder, exist_ok=True)
                generate_and_save_ply(depth_img_path, ply_save_folder)
        else:
            depth_img_path = color_img_path.replace("raw","hole_refined_depth")
            ply_save_folder = os.path.join(self.output_folder, "hole_refined_ply")
            os.makedirs(ply_save_folder, exist_ok=True)
            generate_and_save_ply(depth_img_path, ply_save_folder)

    
    def generate_screenshot_for_pcdMesh(self):
        """
        Call function self.generate_screenshot_for_pcdMesh_oneSample 
            to generate screenshot for all sample under ply_folder
        """
        
        for color_img_path in self.color_img_list:
            self.generate_screenshot_for_pcdMesh_oneSample(color_img_path)
    
    def generate_screenshot_for_pcdMesh_oneSample(self, color_img_path):
        """
        Generate "pcd + mesh"'s screenshot for one sample

        Args:
            self.view_mode : str; "topdow" / "front".

        Output:
            screenshots saved to : os.path.join(ply_folder, "screenshot_{}".format(self.view_mode))
        """
        import open3d as o3d
        def generate_screenshot(depth_img_path):
            # Pack as a function to better support Matterport3d ply generation
            pcd_folder = os.path.join(ply_folder, "pcd")
            mesh_folder = os.path.join(ply_folder, "mesh")
            mirror_info = read_json(color_img_path.replace("raw","img_info").replace(".png",".json"))
            mask_path = color_img_path.replace("raw","instance_mask")

            mirror_mask = cv2.imread(mask_path)
            if len(mirror_info) != (np.unique(mirror_mask).shape[0] - 1):
                self.save_error_raw_name(color_img_path.split("/")[-1].split(".")[0])

            for instance_index in np.unique(np.reshape(mirror_mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                try:
                    instance_tag = "_idx"
                    for i in instance_index:
                        instance_tag += "_{}".format(i)
                    instance_tag = color_img_path.split("/")[-1].split(".")[0] + instance_tag

                    pcd_path = os.path.join(pcd_folder,  "{}.ply".format(instance_tag))
                    mesh_path = os.path.join(mesh_folder,  "{}.ply".format(instance_tag))
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    mirror_plane = o3d.io.read_triangle_mesh(mesh_path)
                    # Screenshots are saved under "mesh_refined_ply" or "hole_refined_ply" folder
                    self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), instance_tag)
                    os.makedirs(self.screenshot_output_folder, exist_ok=True)

                    if self.view_mode == "topdown":
                        self.rotate_pcdMesh_topdown(pcd, mirror_plane)
                    else:
                        self.rotate_pcdMesh_front(pcd, mirror_plane)
                except:
                    self.save_error_raw_name(color_img_path.split("/")[-1].split(".")[0])

        if color_img_path.find("m3d") > 0:
            depth_img_path = rreplace(color_img_path.replace("raw","hole_refined_depth").replace("json","png"),"i","d")
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_screenshot(depth_img_path)

            depth_img_path = rreplace(color_img_path.replace("raw","mesh_refined_depth").replace("json","png"),"i","d")
            ply_folder = os.path.join(self.output_folder, "mesh_refined_ply")
            generate_screenshot(depth_img_path)
            
        else:
            depth_img_path = color_img_path.replace("raw","hole_refined_depth")
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_screenshot(depth_img_path)

    def rotate_pcdMesh_topdown(self, pcd, plane):
        """
        Rotate the "pcd + mesh" by topdown view

        Args:
            pcd : Input point cloud.
            plane : Input mesh plane.
        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
                          self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode)).
        """
        import open3d as o3d
        
        screenshot_id = 0
        mesh_center = np.mean(np.array(plane.vertices), axis=0)
        rotation_step_degree = 10
        start_rotation = get_extrinsic(90,0,0,[0,0,0])
        if self.is_matterport3d:
            stage_tranlation = get_extrinsic(0,0,0,[-mesh_center[0],-mesh_center[1] + 9000,-mesh_center[2]])
        else:
            stage_tranlation = get_extrinsic(0,0,0,[-mesh_center[0],-mesh_center[1] + 3000,-mesh_center[2]])
        start_position = np.dot(start_rotation, stage_tranlation)
        def rotate_view(vis):
            
            nonlocal screenshot_id
            T_rotate = get_extrinsic(0,rotation_step_degree*(screenshot_id+1),0,[0,0,0])
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.dot(np.dot(start_rotation, T_rotate), stage_tranlation)
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            
            screenshot_id += 1
            screenshot_save_path = os.path.join(self.screenshot_output_folder, "{0:05d}.png".format(screenshot_id))
            vis.capture_screen_image(filename=screenshot_save_path, do_render=True)
            print("image saved to {}".format(screenshot_save_path))
            if screenshot_id > (360/rotation_step_degree):
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view)
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = start_position
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        vis.run()

    def set_output_folder(self, output_folder):
        self.output_folder = output_folder

    def set_view_mode(self, view_mode):
        """Function to save the view mode"""
        self.view_mode = view_mode

    def rotate_pcdMesh_front(self, pcd, plane):
        """
        Rotate the "pcd + mesh" by front view

        Args:
            pcd : Input point cloud.
            plane : Input mesh plane.

        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
                          self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), instance_tag)
        """
        import open3d as o3d
        
        screenshot_id = 0
        mesh_center = np.mean(np.array(plane.vertices), axis=0)
        rotation_step_degree = 10
        start_position = get_extrinsic(0,0,0,[0,0,3000])

        def rotate_view(vis):
            
            nonlocal screenshot_id
            T_to_center = get_extrinsic(0,0,0,mesh_center)
            T_rotate = get_extrinsic(0,rotation_step_degree*(screenshot_id+1),0,[0,0,0])
            T_to_mesh = get_extrinsic(0,0,0,-mesh_center)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = np.dot(start_position, np.dot(np.dot(T_to_center, T_rotate),T_to_mesh))
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            
            screenshot_id += 1
            screenshot_save_path = os.path.join(self.screenshot_output_folder, "{0:05d}.png".format(screenshot_id))
            vis.capture_screen_image(filename=screenshot_save_path, do_render=True)
            print("image saved to {}".format(screenshot_save_path))
            if screenshot_id > (360/rotation_step_degree):
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view)
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam.extrinsic = start_position
        vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        vis.run()

    def generate_video_for_all(self):
        """
        Call function self.generate_one_video_ffmpeg 
            to generate video for all sample under ply_folder/screenshot_[view_mode]
        """
        for color_img_path in self.color_img_list:
            self.generate_one_video_ffmpeg(color_img_path)

    def generate_one_video_ffmpeg(self, color_img_path):
        """
        Generate video for one sample 

        Args: 
            color_img_path : the color image path of the sample
        
        Output: 
            .mp4 under video_saved_folder
        """
        def generate_video_to_call(ply_folder):
            # Pack as a function to better support Matterport3d ply generation
            # Videos are saved under "mesh_refined_ply" or "hole_refined_ply" folder
            video_saved_folder = os.path.join(ply_folder, "video_{}".format(self.view_mode))
            os.makedirs(video_saved_folder, exist_ok=True)
            img_info = color_img_path.replace("raw", "img_info").replace("png", "json")
            mirror_info = read_json(img_info)

            for item in mirror_info.items():
                id = item[0]
                instance_tag = color_img_path.split("/")[-1].split(".")[0] + "_idx_" + id
                one_screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), instance_tag)
                one_video_save_path = os.path.join(video_saved_folder, "{}.mp4".format(instance_tag))
                if not os.path.exists(one_screenshot_output_folder):
                    print("{} path not exists!")
                    return
                try:
                    start_time = time.time()
                    if os.path.exists(one_video_save_path):
                        if not self.overwrite:
                            print("{} video exists!".format(one_video_save_path))
                            continue
                        else:
                            os.remove(one_video_save_path)
                    command = "ffmpeg -f image2 -i " + one_screenshot_output_folder + "/%05d.png " + one_video_save_path
                    os.system(command)
                    print("video saved to {}, used time :{}".format(one_video_save_path, time.time() - start_time))
                    start_time = time.time()
                except:
                    self.save_error_raw_name(color_img_path.split("/")[-1].split(".")[0])

        if color_img_path.find("m3d") > 0:
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_video_to_call(ply_folder)
            ply_folder = os.path.join(self.output_folder, "mesh_refined_ply")
            generate_video_to_call(ply_folder)
        else:
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_video_to_call(ply_folder)
        


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="6")
    parser.add_argument(
        '--data_main_folder', default="/project/3dlg-hcvc/mirrors/data/nyu/final_data/precise")
    parser.add_argument(
        '--process_index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    parser.add_argument('--overwrite', help='overwrite files under --output_folder or not',action='store_true')
    parser.add_argument(
        '--f', default=519, type=int, help="camera focal length")
    parser.add_argument(
        '--window_w', default=800, type=int, help="width of the visilization window")
    parser.add_argument(
        '--window_h', default=800, type=int, help="height of the visilization window")
    parser.add_argument(
        '--output_folder', default="/project/3dlg-hcvc/mirrors/www/final_verification/nyu")
    parser.add_argument(
        '--view_mode', default="front", help="object view angle : (1) topdown (2) front")
    args = parser.parse_args()

    vis_tool = Dataset_visulization(data_main_folder=args.data_main_folder, process_index=args.process_index, \
                                    multi_processing=args.multi_processing, f=args.f, \
                                    output_folder = args.output_folder, overwrite=args.overwrite, \
                                    window_w=args.window_w, window_h=args.window_h, view_mode=args.view_mode)
    if args.stage == "1":
        vis_tool.generate_pcdMesh_for_whole_dataset()
    elif args.stage == "2":
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_screenshot_for_pcdMesh()
        vis_tool.set_view_mode("front")
        vis_tool.generate_screenshot_for_pcdMesh()
    elif args.stage == "3":
        vis_tool.generate_screenshot_for_pcdMesh()
    elif args.stage == "4":
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_video_for_all()
        vis_tool.set_view_mode("front")
        vis_tool.generate_video_for_all()
    elif args.stage == "5":
        vis_tool.generate_video_for_all()
    elif args.stage == "6":
        vis_tool.generate_colored_depth_for_whole_dataset()
    elif args.stage == "all":
        # Generate pcdMesh for visualization
        vis_tool.generate_pcdMesh_for_whole_dataset()
        # Generate colored GT depth map
        vis_tool.generate_colored_depth_for_whole_dataset()
        # Generate screenshot for visualization
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_screenshot_for_pcdMesh()
        vis_tool.set_view_mode("front")
        vis_tool.generate_screenshot_for_pcdMesh()
        # Generate video for visualization
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_video_for_all()
        vis_tool.set_view_mode("front")
        vis_tool.generate_video_for_all()
