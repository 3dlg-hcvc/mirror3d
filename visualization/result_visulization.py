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
from dataset_visualization import *
 

class Result_visulization(Dataset_visulization):

    def generate_pcd_for_whole_dataset(self):
        """
        Call function self.generate_pcd_for_one_GTsample 
            to generate mesh.ply and pcd.ply for all sample under self.data_main_folders
        """
        for one_color_img_path in self.color_img_list:
            self.generate_pcd_for_one_GTsample(one_color_img_path)


    def generate_pcd_for_one_GTsample(self, color_img_path):
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
                pcd = get_pcd_from_rgbd(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)

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

    
    def generate_screenshot_for_pcd(self):
        """
        Call function self.generate_screenshot_for_pcd_oneSample 
            to generate screenshot for all sample under ply_folder
        """
        for color_img_path in self.color_img_list:
            self.generate_screenshot_for_pcd_oneSample(color_img_path)
    
    def generate_screenshot_for_pcd_oneSample(self, color_img_path):
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
                self.save_error_raw_name(color_img_path.split(color_img_path.split("/")[-1]))

            for instance_index in np.unique(np.reshape(mirror_mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                # try:
                instance_tag = "_idx"
                for i in instance_index:
                    instance_tag += "_{}".format(i)
                instance_tag = color_img_path.split("/")[-1].split(".")[0] + instance_tag

                pcd_path = os.path.join(pcd_folder,  "{}.ply".format(instance_tag))
                mesh_path = os.path.join(mesh_folder,  "{}.ply".format(instance_tag))
                pcd = o3d.io.read_point_cloud(pcd_path)
                mirror_plane = o3d.io.read_triangle_mesh(mesh_path)

                self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode), instance_tag)
                os.makedirs(self.screenshot_output_folder, exist_ok=True)

                if self.view_mode == "topdpwn":
                    # Get mirror plane's center coordinate 
                    h, w = instance_mask.shape
                    py = np.where(instance_mask)[0].mean()
                    px = np.where(instance_mask)[1].mean()
                    z0 = depth_map[int(py)][int(px)]
                    x0 = (px - w/2) * (z0/ self.f)
                    y0 = (py- h/2) * (z0/ self.f)
                    self.rotate_pcd_topdown(pcd, mirror_plane, x0, y0, z0)
                else:
                    self.rotate_pcd_front(pcd, mirror_plane)
                # except:
                #     self.save_error_raw_name(color_img_path.split("/")[-1])

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


    def rotate_pcd_topdown(self, pcd, plane, x0, y0, z0):
        """
        Rotate the "pcd + mesh" by topdown view

        Args:
            pcd : Input point cloud.
            plane : Input mesh plane.
            x0, y0, z0 : Input mesh plane center coordinate.

        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
                          self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode)).
        """
        pcd.translate(-np.array(plane.vertices).mean(0), relative=True)
        plane.translate(-np.array(plane.vertices).mean(0), relative=True)
        pcd.rotate(get_3_3_rotation_matrix(90, 0, 0),center=False) 
        plane.rotate(get_3_3_rotation_matrix(90, 0, 0),center=False) 
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
            plane.rotate(object_rotation_matrix,center=False) 
            vis.update_geometry(pcd)
            vis.update_geometry(plane)
            if screenshot_id >= 72:
                vis.destroy_window()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_animation_callback(rotate_view) # TODO
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
        vis.run()

    def set_view_mode(self, view_mode):
        self.view_mode = view_mode

    def rotate_pcd_front(self, pcd, plane):
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
        index = 0
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.window_w,height=self.window_h)
        vis.add_geometry(pcd)
        vis.add_geometry(plane)
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
        def generate_video_to_call(ply_folder):
            # Pack as a function to better support Matterport3d ply generation
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
                    self.save_error_raw_name(color_img_path.split(color_img_path.split("/")[-1]))

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
        '--stage', default="4")
    parser.add_argument(
        '--data_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
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

    vis_tool = Result_visulization(data_main_folder=args.data_main_folder, process_index=args.process_index, \
                                    multi_processing=args.multi_processing, f=args.f, \
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