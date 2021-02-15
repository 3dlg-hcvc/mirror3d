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

    def __init__(self, dataset_main_folder=None, prediction_output_folder = None, method_tag="mirror3D", process_index=0, multi_processing=False, 
                f=519, output_folder=None, overwrite=True, window_w=800, window_h=800, view_mode="topdown"):
        """
        Initilization

        Args:
            dataset_main_folder : Folder raw, hole_raw_depth/ mesh_raw_depth, instance_mask saved folder.
            output_folder(optional) : Inital pcd, img_info, border_vis saved forder (default : dataset_main_folder).
            process_index : The process index of multi_processing.
            multi_processing : Use multi_processing or not (bool).
            border_width : Half of mirror 2D border width (half of cv2.dilate kernel size; 
                           default kernel anchor is at the center); default : 50 --> actualy border width = 25.
            f : Camera focal length of current input data.
        """

        self.dataset_main_folder = dataset_main_folder
        self.method_tag = method_tag
        self.prediction_output_folder = prediction_output_folder
        assert os.path.exists(dataset_main_folder), "please input a valid folder path"
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.overwrite = overwrite
        self.window_w = window_w
        self.window_h = window_h
        self.view_mode = view_mode
        
        if "m3d" not in self.dataset_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.color_img_list = [os.path.join(self.dataset_main_folder, "raw", i) for i in os.listdir(os.path.join(self.dataset_main_folder, "raw"))]
        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.f = f
        if output_folder == None or not os.path.exists(output_folder):
            self.output_folder = self.dataset_main_folder
            print("########## NOTE output saved to {}, this may overwrite your current information ############".format(self.output_folder))
        else:
            self.output_folder = output_folder
        self.error_info_path = os.path.join(self.output_folder, "error_img_list.txt")
    
    def generate_pcd_for_whole_dataset(self):
        """
        Call function self.generate_pcd_for_one_prediction 
            to generate mesh.ply and pcd.ply for all sample under self.dataset_main_folders
        """
        for one_color_img_path in self.color_img_list:
            self.generate_pcd_for_one_prediction(one_color_img_path)

    def generate_pcd_for_one_prediction(self, color_img_path):
        """
        Generate point cloud for specific prediction

        Args:
            color_img_path : The color image absolute path for the specific sample.
        
        Output:
            "point cloud" : Saved under self.output_folder.
        """

        import open3d as o3d
        pcd_save_folder = os.path.join(self.output_folder, self.method_tag, "pred_depth_ply")
        os.makedirs(pcd_save_folder, exist_ok=True)

        sample_name = color_img_path.split("/")[-1]
        pcd_save_path = os.path.join(pcd_save_folder,  sample_name.replace("png","ply"))

        if self.is_matterport3d:
            pred_depth_img_path = os.path.join(self.prediction_output_folder, rreplace(sample_name.replace("raw","pred_depth"), "i", "d"))
        else:
            pred_depth_img_path =  os.path.join(self.prediction_output_folder, sample_name.replace("raw","pred_depth"))

        # Get and save pcd for the instance
        pcd = get_pcd_from_rgbd(self.f, pred_depth_img_path, color_img_path)
        o3d.io.write_point_cloud(pcd_save_path, pcd)
        print("point cloud saved  to :", os.path.abspath(pcd_save_path))


    def generate_screenshot_for_pcd(self):
        """
        Call function self.generate_screenshot_for_pcd_oneSample 
            to generate screenshot for all sample under ply_folder
        """
        for color_img_path in self.color_img_list:
            self.generate_screenshot_for_pcd_oneSample(color_img_path)
    
    def generate_screenshot_for_pcd_oneSample(self, color_img_path):
        """
        Generate pcd screenshot for one predicted depth

        Args:
            self.view_mode : str; "topdow" / "front".

        Output:
            screenshots saved to : self.screenshot_output_folder = os.path.join(self.output_folder, "screenshot_{}".format(self.view_mode), sample_name.split(".")[0])
        """

        import open3d as o3d
        pcd_folder = os.path.join(self.output_folder, self.method_tag, "pred_depth_ply")
        os.makedirs(pcd_folder, exist_ok=True)

        sample_name = color_img_path.split("/")[-1]
        pcd_path = os.path.join(pcd_folder,  sample_name.replace("png","ply"))
        pcd = o3d.io.read_point_cloud(pcd_path)

        if self.is_matterport3d:
            pred_depth_img_path = os.path.join(self.prediction_output_folder, rreplace(sample_name.replace("raw","pred_depth"), "i", "d"))
        else:
            pred_depth_img_path =  os.path.join(self.prediction_output_folder, sample_name.replace("raw","pred_depth"))

        self.screenshot_output_folder = os.path.join(self.output_folder, "screenshot_{}".format(self.view_mode), sample_name.split(".")[0])
        os.makedirs(self.screenshot_output_folder, exist_ok=True)

        if self.view_mode == "topdown":
            self.rotate_pcd_topdown(pcd)
        else:
            self.rotate_pcd_front(pcd)


    def rotate_pcd_topdown(self, pcd):
        """
        Rotate the "pcd + mesh" by topdown view

        Args:
            pcd : Input point cloud.

        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
        """
        pcd.rotate(get_3_3_rotation_matrix(90, 0, 0),center=False) 
        object_rotation_matrix = get_3_3_rotation_matrix(0, 0, 10)
        screenshot_id = 0

        def rotate_view(vis):
            nonlocal screenshot_id
            nonlocal pcd
            nonlocal object_rotation_matrix
            screenshot_id += 1
            screenshot_save_path = os.path.join(self.screenshot_output_folder, "{0:05d}.png".format(screenshot_id))
            time.sleep(0.05)
            vis.capture_screen_image(screenshot_save_path)
            print("image saved to {}".format(screenshot_save_path))
            pcd.rotate(object_rotation_matrix,center=False) 
            vis.update_geometry(pcd)
            if screenshot_id >= 72:
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view) # TODO
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        vis.run()

    def set_view_mode(self, view_mode):
        """Function to save the view mode"""
        self.view_mode = view_mode

    def rotate_pcd_front(self, pcd):
        """
        Rotate the "pcd + mesh" by front view

        Args:
            pcd : Input point cloud.

        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
        """
        import open3d as o3d
        index = 0
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        ctrl = vis.get_view_control()
        ctrl.rotate(0, 1000)
        while vis.poll_events():
            index += 1
            if index%4 == 0:
                screenshot_save_path = os.path.join(self.screenshot_output_folder, "{0:05d}.png".format(int(index/4)))
                time.sleep(0.05)
                vis.capture_screen_image(screenshot_save_path)
                print("image saved to {}".format(screenshot_save_path))
            ctrl.rotate(10, 0)
            vis.update_renderer()
            if index > 300:
                vis.destroy_window()
                break

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

        ply_folder = os.path.join(self.output_folder, self.method_tag, "pred_depth_ply")
        # Pack as a function to better support Matterport3d ply generation
        video_saved_folder = os.path.join(ply_folder, "video_{}".format(self.view_mode))
        os.makedirs(video_saved_folder, exist_ok=True)
        img_info = color_img_path.replace("raw", "img_info").replace("png", "json")
        sample_tag = color_img_path.split("/")[-1].split(".")[0]

        one_screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), sample_tag)
        one_video_save_path = os.path.join(video_saved_folder, "{}.mp4".format(sample_tag))
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
            self.save_error_raw_name(color_img_path.split(color_img_path.split("/")[-1]))


    def gen_GT_errorMap_pred_topdownScreenshot_one_sample(self, color_img_path):
        """
        Args:
            self.prediction_output_folder
            self.dataset_main_folder
            self.method_tag

        Output:
            (1) colored GT depth map
            (2) colored RMSE error map
            (3) colored predicted depth map
            (4) topdown view of the corresponding predicted pcd
        """
        
        ply_folder = os.path.join(self.output_folder, self.method_tag, "pred_depth_ply")
        # Pack as a function to better support Matterport3d ply generation
        video_saved_folder = os.path.join(ply_folder, "video_{}".format(self.view_mode))
        os.makedirs(video_saved_folder, exist_ok=True)
        img_info = color_img_path.replace("raw", "img_info").replace("png", "json")
        sample_tag = color_img_path.split("/")[-1].split(".")[0]
        sample_name = color_img_path.split("/")[-1]
        
        one_screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), sample_tag)
        topdown_screenshot_path = os.path.join(one_screenshot_output_folder, "00001.png")

        if self.is_matterport3d:
            pred_depth_img_path = os.path.join(self.prediction_output_folder, rreplace(sample_name.replace("raw","pred_depth"), "i", "d"))
            gt_depth_img_path = rreplace(color_img_path.replace("raw","pred_depth"), "i", "d")) 
        else:
            pred_depth_img_path =  os.path.join(self.prediction_output_folder, sample_name.replace("raw","pred_depth"))
            gt_depth_img_path = color_img_path.replace("raw","pred_depth")

        gt_depth = cv2.imread(gt_depth_img_path, cv2.IMREAD_ANYDEPTH)
        pred_depth = cv2.imread(pred_depth_img_path, cv2.IMREAD_ANYDEPTH)
        rmse = (gt_depth - pred_depth) ** 2

        
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="4")
    parser.add_argument(
        '--prediction_output_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
    parser.add_argument(
        '--method_tag', default="mirror3D")
    parser.add_argument(
        '--dataset_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
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
        '--output_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
    parser.add_argument(
        '--view_mode', default="topdown", help="object view angle : (1) topdown (2) front")
    args = parser.parse_args()

    vis_tool = Dataset_visulization(dataset_main_folder=args.dataset_main_folder, process_index=args.process_index, \
                                    multi_processing=args.multi_processing, f=args.f, \
                                    prediction_output_folder = args.prediction_output_folder, method_tag=args.method_tag \
                                    output_folder=args.output_folder, overwrite=args.overwrite, \
                                    window_w=args.window_w, window_h=args.window_h, view_mode=args.view_mode)
    if args.stage == "1":
        vis_tool.generate_pcd_for_whole_dataset()
    elif args.stage == "2":
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_screenshot_for_pcd()
        vis_tool.set_view_mode("front")
        vis_tool.generate_screenshot_for_pcd()
    elif args.stage == "3":
        vis_tool.generate_screenshot_for_pcd()
    elif args.stage == "4":
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_video_for_all()
        vis_tool.set_view_mode("front")
        vis_tool.generate_video_for_all()
    elif args.stage == "5":
        vis_tool.generate_video_for_all()
    elif args.stage == "all":
        # Generate pcd for visualization
        vis_tool.generate_pcd_for_whole_dataset()
        # Generate screenshot for visualization
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_screenshot_for_pcd()
        vis_tool.set_view_mode("front")
        vis_tool.generate_screenshot_for_pcd()
        # Generate video for visualization
        vis_tool.set_view_mode("topdown")
        vis_tool.generate_video_for_all()
        vis_tool.set_view_mode("front")
        vis_tool.generate_video_for_all()
