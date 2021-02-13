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
 

class Data_visulization(Plane_annotation_tool):

    def __init__(self, data_main_folder=None, process_index=0, multi_processing=False, f=519, output_folder=None):
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
            self.generate_pcdMesh_for_one_GTsample(color_img_path)


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
            smaple_name = color_img_path.split("/")[-1]
            img_info_path = color_img_path.replace("raw","img_info").replace("png","json")

            one_img_info = read_json(img_info_path)
            
            #  Get pcd and masked RGB image for each instance
            for instance_index in np.unique(np.reshape(mask,(-1,3)), axis = 0):
                if sum(instance_index) == 0: # background
                    continue

                instance_tag = "_idx"
                for i in instance_index:
                    instance_tag += "_{}".format(i)
                instance_tag = smaple_name + instance_tag
                

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
    plane_anno_tool = Data_visulization(args.data_main_folder, args.index, False)
    plane_anno_tool.generate_pcdMesh_for_one_GTsample("/Users/tanjiaqi/Desktop/SFU/mirror3D/test/raw/150.png")