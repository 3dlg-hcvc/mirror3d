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
        self.error_info_path = os.path.join(self.output_folder, "error_img_list.txt")
    

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
                instance_tag = color_img_path.split("/")[-1] + instance_tag
                binary_instance_mask = get_grayscale_instanceMask(mask, instance_index)
                plane_parameter = one_img_info[instance_tag.split("_idx_")[1]]

                # Get pcd for the instance
                pcd = get_pcd_from_rgbd(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)

                # Get mirror plane for the instance
                mirror_points = get_points_in_mask(self.f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.stack(mirror_points,axis=0)))
                mirror_plane = get_mirror_init_plane_from_mirrorbbox(plane_parameter, mirror_bbox)

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

    
    def generate_screenshot_for_pcdMesh():
        """
        Call function self.generate_screenshot_for_pcdMesh_oneSample 
            to generate screenshot for all sample under ply_folder
        """
        for color_img_path in self.color_img_list:
            self.generate_pcdMesh_for_one_GTsample(color_img_path)
    
    def generate_screenshot_for_pcdMesh_oneSample(color_img_path):
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
            mirror_info = read_json(img_info_path.replace("raw","img_info"))
            mask_path = img_info_path.replace("raw","instance_mask")

            mirror_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            if len(mirror_info) != (np.unique(mirror_mask).shape[0] - 1):
                self.save_error_raw_name(color_img_path.split(color_img_path.split("/")[-1]))

            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue
                try:
                    instance_tag = "_idx"
                    for i in instance_index:
                        instance_tag += "_{}".format(i)
                    instance_tag = color_img_path.split("/")[-1] + instance_tag

                    pcd_path = os.path.join(pcd_folder,  "{}.ply".format(instance_tag))
                    mesh_path = os.path.join(mesh_folder,  "{}.ply".format(instance_tag))
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    mirror_plane = o3d.io.read_triangle_mesh(mesh_path)

                    self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode))
                    os.makedirs(self.screenshot_output_folder, exist_ok=True)

                    if self.view_mode == "topdpwn":
                        # Get mirror plane's center coordinate 
                        h, w = instance_mask.shape
                        py = np.where(instance_mask)[0].mean()
                        px = np.where(instance_mask)[1].mean()
                        z0 = depth_map[int(py)][int(px)]
                        x0 = (px - w/2) * (z0/ self.f)
                        y0 = (py- h/2) * (z0/ self.f)
                        self.rotate_pcdMesh_topdown(pcd, plane, x0, y0, z0)
                    else:
                        self.rotate_pcdMesh_front(pcd, plane)
                except:
                    self.save_error_raw_name(color_img_path.split(color_img_path.split("/")[-1]))

        if color_img_path.find("m3d") > 0:
            depth_img_path = rreplace(img_info_path.replace("raw","hole_refined_depth").replace("json","png"),"i","d")
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_screenshot(depth_img_path)

            depth_img_path = rreplace(img_info_path.replace("raw","mesh_refined_depth").replace("json","png"),"i","d")
            ply_folder = os.path.join(self.output_folder, "mesh_refined_ply")
            generate_screenshot(depth_img_path)
            
        else:
            depth_img_path = img_info_path.replace("img_info","hole_refined_depth").replace("json","png")
            ply_folder = os.path.join(self.output_folder, "hole_refined_ply")
            generate_screenshot(depth_img_path)


    def rotate_pcdMesh_topdown(self, pcd, plane, x0, y0, z0):
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


    def rotate_pcdMesh_front(self, pcd, plane):
        """
        Rotate the "pcd + mesh" by front view

        Args:
            pcd : Input point cloud.
            plane : Input mesh plane.

        Output:
            Screenshots : Saved under output folder (self.screenshot_output_folder);
                          self.screenshot_output_folder = os.path.join(ply_folder, "screenshot_{}".format(self.view_mode)).
        """
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

    def generate_one_video_ffmpeg(self, color_img_path):
        img_info = color_img_path.replace("raw", "img_info").replace("png", "json")
        mirror_info = read_json(img_info)

        # for item in mirror_info.items():
        #     id = item[0]
        #     try:
        #         one_screenshot_output_folder = self.screenshot_output_folder + "_" + str(id)
        #         start_time = time.time()
        #         one_video_save_path = one_screenshot_output_folder.replace("screenshot","video") + ".mp4"
        #         one_video_save_folder = os.path.split(one_video_save_path)[0]
        #         os.makedirs(one_video_save_folder, exist_ok=True)
        #         if os.path.exists(one_video_save_path):
        #             if not self.overwrite:
        #                 print("{} video exists!".format(one_video_save_path))
        #                 continue
        #             else:
        #                 os.remove(one_video_save_path)
        #         command = "ffmpeg -f image2 -i " + one_screenshot_output_folder + "/%05d.png " + one_video_save_path
        #         os.system(command)
        #         print("video saved to {}, used time :{}".format(one_video_save_path, time.time() - start_time))
        #         start_time = time.time()
        #     except:
        #         self.note_error(img_info.replace(self.data_folder_path,""))

    
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
    plane_anno_tool = Dataset_visulization(args.data_main_folder, args.index, False)
    plane_anno_tool.generate_pcdMesh_for_whole_dataset()
    # plane_anno_tool.generate_pcdMesh_for_whole_dataset("/Users/tanjiaqi/Desktop/SFU/mirror3D/test/raw/150.png")