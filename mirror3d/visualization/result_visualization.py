import numpy as np
import cv2
import argparse
import os
import bs4
from mirror3d.utils.algorithm import *
from mirror3d.utils.general_utils import *
from mirror3d.utils.plane_pcd_utils import *
from mirror3d.annotation.plane_annotation.plane_annotation_tool import *


class ResultVisualization(PlaneAnnotationTool):

    def generate_pcd_for_vis(self, input_txt):
        """
        Output:
            "point cloud": Saved under self.output_folder.
        """

        import open3d as o3d
        # Pack as a function to better support Matterport3d ply generation
        def generate_and_save_ply(color_img_path, depth_img_path, mask_img_path, pcd_save_folder, f):
            os.makedirs(pcd_save_folder, exist_ok=True)

            mask = cv2.imread(mask_img_path, cv2.IMREAD_ANYDEPTH)
            #  Get pcd and masked RGB image for each instance
            for instance_index in np.unique(mask):
                if instance_index == 0:  # background
                    continue
                save_name = color_img_path.split("/")[-1].split(".")[0] + "_idx_" + str(instance_index)
                pcd_save_path = os.path.join(pcd_save_folder, "{}.ply".format(save_name))
                binary_instance_mask = (mask == instance_index).astype(np.uint8)

                if os.path.exists(pcd_save_path) and not self.overwrite:
                    print(pcd_save_path, "exist! continue")
                    return

                # Get pcd for the instance
                pcd = get_pcd_from_rgbd_depthPath(f, depth_img_path, color_img_path, mirror_mask=binary_instance_mask)

                # Get mirror plane for the instance
                mirror_points = get_points_in_mask(f, depth_img_path, mirror_mask=binary_instance_mask)
                mirror_pcd = o3d.geometry.PointCloud()
                mirror_pcd.points = o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0))
                mirror_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(np.stack(mirror_points, axis=0)))

                o3d.io.write_point_cloud(pcd_save_path, pcd)
                print("point cloud saved  to :", os.path.abspath(pcd_save_path))


        process_list = self.get_list_to_process(read_txt(input_txt))
        for item in process_list:
            if len(item.strip().split()) != 5:
                continue
            color_img_path, depth_img_path, mask_img_path, pcd_save_folder, f = item.strip().split()
            if not os.path.exists(color_img_path) or not os.path.exists(depth_img_path):
                print("invalid line : ", item)
                print("input txt format: [input color image path] [input depth image path] [input integer mask path] "
                      "[folder to save the output pointcloud] [focal length of this sample]")
                continue
            f = self.get_and_check_focal_length(f, item)
            generate_and_save_ply(color_img_path, depth_img_path, mask_img_path, pcd_save_folder, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--function', default="1")
    parser.add_argument(
        '--input_txt', default="")
    parser.add_argument('--multi_processing', help='do multi-process or not', action='store_true')
    parser.add_argument('--overwrite', help='overwrite current result or not', action='store_true')
    parser.add_argument(
        '--process_index', default=0, type=int, help="if do --multi_processing please input the process index")
    parser.add_argument(
        '--above_height', default=3000, type=int, help="camera height to the mirror plane center in the topdown view")
    parser.add_argument(
        '--video_num_per_page', default=100, type=int)
    parser.add_argument(
        '--html_output_folder', default="")
    args = parser.parse_args()

    result_vis = ResultVisualization(process_index=args.process_index, multi_processing=args.multi_processing,
                                          overwrite=args.overwrite)

    if args.function == "1":
        print("input txt format: [input color image path] [input depth image path] [input integer mask path] "
                      "[folder to save the output pointcloud] [focal length of this sample]")
        result_vis.generate_pcd_for_vis(args.input_txt)
    elif args.function == "2":
        print("input txt format: [path to pointcloud] [path to mesh plane] [screenshot output main folder]")
        result_vis.set_view_mode(args.view_mode)
        result_vis.generate_video_screenshot_from_3Dobject(args.input_txt, args.above_height)
    elif args.function == "3":
        print("input txt format: [input depth image path] [colored depth map saved path]")
        result_vis.gen_colored_grayscale_for_depth(args.input_txt)
    elif args.function == "4":
        print("input txt format: [sample id] [input color image path] [colored depth map saved path] [front view "
              "video path] [topdown view video path]")
        result_vis.gen_verification_html(args.input_txt, args.video_num_per_page, args.html_output_folder)
