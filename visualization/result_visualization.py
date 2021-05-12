import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
import sys
import bs4
from utils.algorithm import *
from utils.general_utlis import *
from utils.plane_pcd_utils import *
import json
import shutil
from annotation.plane_annotation.plane_annotation_tool import *
from visualization.dataset_visualization import Dataset_visulization
from tqdm import tqdm




class Dataset_visulization(Dataset_visulization):

    def __init__(self, pred_w=480, pred_h=640, dataset_main_folder=None, test_json="", method_tag="mirror3D", process_index=0, multi_processing=False, 
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
        assert os.path.exists(dataset_main_folder), "please input a valid folder path"
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.overwrite = overwrite
        self.window_w = window_w
        self.window_h = window_h
        self.view_mode = view_mode
        self.pred_w = pred_w
        self.pred_h = pred_h
        
        if "m3d" not in self.dataset_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.color_img_list = []
        input_images = read_json(test_json)["images"]
        for one_info in input_images: 
            self.color_img_list.append(os.path.join(dataset_main_folder, one_info["img_path"]))
        self.color_img_list.sort()

        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.f = f
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.error_info_path = os.path.join(self.output_folder, "error_img_list.txt")
    
    def generate_method_pred(self, info_txt_path):
        self.method_predFolder = dict()
        for one_line in read_txt(info_txt_path):
            one_info = one_line.strip().split()
            method_tag = one_info[0]
            pred_folder = one_info[1]
            self.method_predFolder[method_tag] = pred_folder
    
    def generate_color_depth_for_all_pred(self):

        for one_color_img_path in self.color_img_list:
            self.generate_color_depth_for_one_pred(one_color_img_path)

    def generate_color_depth_for_one_pred(self, color_img_path):
        """
        Generate point cloud for specific prediction

        Args:
            color_img_path : The color image absolute path for the specific sample.
        
        Output:
            "point cloud" : Saved under self.output_folder.
        """
        import open3d as o3d
        for item in self.method_predFolder.items():
            method_tag = item[0]
            prediction_output_folder = item[1]

            one_colored_pred_depth_folder =  os.path.join(self.output_folder, method_tag, "colored_pred_depth")
            os.makedirs(one_colored_pred_depth_folder, exist_ok=True)
            one_colored_pred_error_map_folder =  os.path.join(self.output_folder, method_tag, "colored_pred_error_map")
            os.makedirs(one_colored_pred_error_map_folder, exist_ok=True)

            one_info_folder =  os.path.join(self.output_folder, method_tag, "info")
            os.makedirs(one_info_folder, exist_ok=True)

            sample_name = color_img_path.split("/")[-1]

            if self.is_matterport3d:
                pred_depth_img_path = os.path.join(prediction_output_folder, rreplace(sample_name, "i", "d"))
                colored_pred_depth_save_path = os.path.join(one_colored_pred_depth_folder,  rreplace(sample_name, "i", "d"))
                colored_pred_error_map_save_path = os.path.join(one_colored_pred_error_map_folder,  rreplace(sample_name, "i", "d"))
                gt_depth_img_path = rreplace(color_img_path.replace("raw","mesh_refined_depth"),"i", "d")
            else:
                pred_depth_img_path =  os.path.join(prediction_output_folder, sample_name)
                colored_pred_depth_save_path = os.path.join(one_colored_pred_depth_folder,  sample_name)
                colored_pred_error_map_save_path = os.path.join(one_colored_pred_error_map_folder,  sample_name)
                gt_depth_img_path = color_img_path.replace("raw","hole_refined_depth")
            
            info_save_path = os.path.join(one_info_folder, "{}.json".format(sample_name.split(".")[0]))
            gt_depth = cv2.imread(gt_depth_img_path, cv2.IMREAD_ANYDEPTH)
            pred_depth = cv2.imread(pred_depth_img_path, cv2.IMREAD_ANYDEPTH)
            if self.is_matterport3d:
                depth_shift = 4000
            else:
                depth_shift = 1000
            gt_depth = np.asarray(cv2.resize(gt_depth, dsize=(self.pred_w, self.pred_h), interpolation=cv2.INTER_NEAREST), dtype=np.float32) / depth_shift
            pred_depth = np.asarray(cv2.resize(pred_depth, dsize=(self.pred_w, self.pred_h), interpolation=cv2.INTER_NEAREST), dtype=np.float32) / depth_shift
            
            rmse = (gt_depth - pred_depth) ** 2
            score = float(np.mean(rmse))
            if os.path.exists(info_save_path):
                info = read_json(info_save_path)
            else:
                info = dict()

            info["RMSE"] = score


            save_json(info_save_path, info)
            save_heatmap_no_border(pred_depth, colored_pred_depth_save_path)
            save_heatmap_no_border(rmse, colored_pred_error_map_save_path)

            print("colored_pred_depth_save_path to :", os.path.abspath(colored_pred_depth_save_path))
            print("colored_pred_error_map_save_path to :", os.path.abspath(colored_pred_error_map_save_path))


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

        def save_topdown(pcd, topdown_img_save_path):

            mesh_center = pcd.get_center()
            rotation_step_degree = 10
            start_rotation = get_extrinsic(90,0,0,[0,0,0])
            if self.is_matterport3d:
                stage_tranlation = get_extrinsic(0,0,0,[-mesh_center[0],-mesh_center[1] + 13000,-mesh_center[2]])
            else:
                stage_tranlation = get_extrinsic(0,0,0,[-mesh_center[0],-mesh_center[1] + 8000,-mesh_center[2]])
            start_position = np.dot(start_rotation, stage_tranlation)
            def rotate_view(vis):
                T_rotate = get_extrinsic(0,rotation_step_degree*(1),0,[0,0,0])
                cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam.extrinsic = np.dot(np.dot(start_rotation, T_rotate), stage_tranlation)
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                
                vis.capture_screen_image(filename=topdown_img_save_path, do_render=True)
                print("image saved to {}".format(topdown_img_save_path))
                vis.destroy_window()

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_animation_callback(rotate_view)
            vis.create_window(width=self.window_w,height=self.window_h)
            vis.get_render_option().point_size = 1.0
            vis.add_geometry(pcd)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = start_position
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            vis.run()


        def save_front(pcd, front_img_save_path):
            
            mesh_center = pcd.get_center()
            rotation_step_degree = 10
            start_position = get_extrinsic(0,0,0,[0,0,3000])

            def rotate_view(vis):
                T_to_center = get_extrinsic(0,0,0,mesh_center)
                T_rotate = get_extrinsic(0,rotation_step_degree*(1),0,[0,0,0])
                T_to_mesh = get_extrinsic(0,0,0,-mesh_center)
                cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam.extrinsic = np.dot(start_position, np.dot(np.dot(T_to_center, T_rotate),T_to_mesh))
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                
                vis.capture_screen_image(filename=front_img_save_path, do_render=True)
                print("image saved to {}".format(front_img_save_path))
                vis.destroy_window()
                return False

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_animation_callback(rotate_view)
            vis.create_window(width=self.window_w,height=self.window_h)
            vis.get_render_option().point_size = 1.0
            vis.add_geometry(pcd)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = start_position
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            vis.run()

        for item in self.method_predFolder.items():
            method_tag = item[0]
            prediction_output_folder = item[1]

            one_method_pcd_save_folder =  os.path.join(self.output_folder, method_tag, "pred_depth_ply")
            os.makedirs(one_method_pcd_save_folder, exist_ok=True)

            sample_name = color_img_path.split("/")[-1]
            pcd_save_path = os.path.join(one_method_pcd_save_folder,  sample_name.replace("png","ply"))
            topdown_img_save_path = os.path.join(one_method_pcd_save_folder,  "topdown_{}".format(sample_name))
            front_img_save_path = os.path.join(one_method_pcd_save_folder,  "front_{}".format(sample_name))


            if self.is_matterport3d:
                pred_depth_img_path = os.path.join(prediction_output_folder, rreplace(sample_name, "i", "d"))
            else:
                pred_depth_img_path =  os.path.join(prediction_output_folder, sample_name)
            # Get and save pcd for the instance
            pcd = get_pcd_from_rgbd_depthPath(f=self.f, depth_img_path=pred_depth_img_path, color_img_path=color_img_path, w=self.pred_w, h=self.pred_h)
            o3d.io.write_point_cloud(pcd_save_path, pcd)
            save_front(pcd, front_img_save_path)
            save_topdown(pcd, topdown_img_save_path)
            print("point cloud saved  to :", os.path.abspath(pcd_save_path))


    def set_view_mode(self, view_mode):
        """Function to save the view mode"""
        self.view_mode = view_mode

                
    def generate_html(self, vis_saved_main_folder="", sample_num_per_page=50, template_path=""):
        """
        (1) under vis_saved_folder there should only be the vislization output sub-folders and nothing else

        Each line will show:
            1. Sample ID & method name 
            2. Color image
            3. Colored RMSE map
            4. Colored predict depth
            5. Front view
            6. Topdown view
        """
        if self.is_matterport3d:
            method_folder_list = ["VNL+Mirror3DNet","VNL-raw","BTS+Mirror3DNet","BTS-raw","VNL-ref","BTS-ref","Mirror3DNet-raw","Mirror3DNet-DE-raw","PlaneRCNN-raw","PlaneRCNN-DE-raw","saic+Mirror3DNet","saic-raw","Mirror3DNet-ref","Mirror3DNet-DE-ref","PlaneRCNN-ref","PlaneRCNN-DE-ref","saic-ref","mesh-D+Mirror3DNet","sensor-D+Mirror3DNet","mesh-D","sensor-D"]
            # method_folder_list = ["GT", "meshD", "meshD_Mirror3DNet", "BTS_refD", "BTS_rawD", "BTS_Mirror3DNet", "VNL_refD", "VNL_rawD", "VNL_Mirror3DNet", "SAIC_refD", "SAIC_rawD", "SAIC_Mirror3DNet"]
        else:
             method_folder_list = ["sensor-D","sensor-D+Mirror3DNet-NYUv2_ft","sensor-D+Mirror3DNet-m3d","saic-ref","PlaneRCNN-ref-NYUv2_ft","PlaneRCNN-ref-m3d","Mirror3DNet-NYUv2_ft-ref","Mirror3DNet-m3d-ref","saic-raw","saic+Mirror3DNet-NYUv2_ft","saic+Mirror3DNet-m3d","PlaneRCNN-raw-NYUv2_ft","PlaneRCNN-raw-m3d","Mirror3DNet-NYUv2_ft-raw","Mirror3DNet-m3d-raw","BTS-ref","VNL-ref","BTS-raw","BTS+Mirror3DNet-NYUv2_ft","BTS+Mirror3DNet-m3d","VNL-raw","VNL+Mirror3DNet-NYUv2_ft","VNL+Mirror3DNet-m3d"]
            # method_folder_list = ["GT", "sensorD", "sensorD_Mirror3DNet", "BTS_refD", "BTS_rawD", "BTS_Mirror3DNet", "VNL_refD", "VNL_rawD", "VNL_Mirror3DNet", "SAIC_refD", "SAIC_rawD", "SAIC_Mirror3DNet"]
        colorImgSubset_list = [self.color_img_list[x:x+sample_num_per_page] for x in range(0, len(self.color_img_list), sample_num_per_page)]

        for html_index, one_colorSubset in enumerate(colorImgSubset_list):
            with open(template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")
            for one_color_img_path in self.color_img_list:
                sample_name = os.path.split(one_color_img_path)[-1]
                sample_id = sample_name.split(".")[0]
                if self.is_matterport3d:
                    one_depth_sample_name = rreplace(sample_name, "i", "d")
                else:
                    one_depth_sample_name = sample_name
                for one_method_name in method_folder_list:
                    one_method_folder_path = os.path.join(vis_saved_main_folder, one_method_name)
                    one_RMSE_map = os.path.join(one_method_folder_path, "colored_pred_error_map", one_depth_sample_name)
                    one_predD_map = os.path.join(one_method_folder_path, "colored_pred_depth", one_depth_sample_name)
                    one_front_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "front_{}".format(sample_name))
                    one_topdown_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "topdown_{}".format(sample_name))

                    new_div = soup.new_tag("div")
                    new_div['class'] = "one-sample"
                    
                    soup.body.append(new_div)

                    one_text = soup.new_tag("div")
                    one_text["class"] = "one-item"
                    if not self.is_matterport3d:
                        one_text['style'] = "padding-top: 100px;font-size: 25pt;"
                    else:
                        one_text['style'] = "padding-top: 100px;"
                    one_text.string = sample_id
                    new_div.append(one_text)

                    one_text = soup.new_tag("div")
                    one_text["class"] = "one-item"
                    if not self.is_matterport3d:
                        one_text['style'] = "padding-top: 100px;font-size: 25pt;"
                    else:
                        one_text['style'] = "padding-top: 100px;"
                    one_text.string = one_method_name
                    new_div.append(one_text)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    color_img_path = os.path.relpath(one_color_img_path, self.output_folder)
                    color_img.append(soup.new_tag('img', src=color_img_path))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_RMSE_map = os.path.relpath(one_RMSE_map, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_RMSE_map))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_predD_map = os.path.relpath(one_predD_map, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_predD_map))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_front_view_img = os.path.relpath(one_front_view_img, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_front_view_img))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_topdown_view_img = os.path.relpath(one_topdown_view_img, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_topdown_view_img))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)
                
            html_path = os.path.join(self.output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)
            
            print("result visulisation saved in link {}".format(html_path.replace("/project/3dlg-hcvc/mirrors/www","http://aspis.cmpt.sfu.ca/projects/mirrors")))

    def gen_latex_table_sep(self, method_predFolder_txt, midrule_index):
        folder_list = [line.strip().split()[-1] for line in read_txt(method_predFolder_txt)]

        main_table_lines = read_txt("./visualization/table_template/main_table_begin.txt")
        sup_table_part1_lines = read_txt("./visualization/table_template/sup1_table_begin.txt")
        sup_table_part2_lines = read_txt("./visualization/table_template/sup2_table_begin.txt")
        for exp_index, one_result_folder in enumerate(folder_list):

            one_line_latex_source_path = os.path.join(one_result_folder, "latex_result.json")
            one_latex_lines = [item[1] for item in read_json(one_line_latex_source_path).items()]
            main_table_lines.append(one_latex_lines[0]+"\\")
            sup_table_part1_lines.append(one_latex_lines[1]+"\\")
            sup_table_part2_lines.append(one_latex_lines[2]+"\\")

            if (exp_index + 1) in midrule_index:
                main_table_lines.append("\midrule")
                sup_table_part1_lines.append("\midrule")
                sup_table_part2_lines.append("\midrule")
        
        main_table_lines.append("\\bottomrule\end{tabular}}\end{table}")
        sup_table_part1_lines.append("\\bottomrule\end{tabular}}\end{table}")
        sup_table_part2_lines.append("\\bottomrule\end{tabular}}\end{table}")

        print(" ##################### main latex table ##################### ")
        for line in main_table_lines:
            print(line)

    def gen_latex_table_with_sample_std(self, method_order_txt, all_info_json, midrule_index):

        def identify_best_for_subLines(sublines, metrics_list):
            downmetrics = ['RMSE', 's-RMSE', 'Rel']
            upmetrics = ['SSIM','$d_{1.05}$','$d_{1.12}$','$d_{1.25}$','$d_{1.25^2','$d_{1.25^3}$']
            col_num = len(sublines[0].split("&")) - 3
            metrics_list = metrics_list[0].split(",")

            subline_score = []
            for one_line in sublines:
                subline_score.append([float(one_score.strip()) for one_score in one_line.replace('\\','').split("&")[3:]])
            subline_score = np.array(subline_score)
            
            # identify the best score in a column
            best_score = np.zeros(col_num)
            for col_index in range(col_num):
                metric = metrics_list[int(col_index/3)]
                if metric in downmetrics:
                    best_score[col_index] = subline_score[:,col_index].min()
                else:
                    best_score[col_index] = subline_score[:,col_index].max()
            
            # replace the best score with \best{}
            best_lines = []
            for one_line in sublines: 
                ori_str_scores = [one_score.strip() for one_score in one_line.replace('\\','').split("&")[3:]]
                one_best_line = one_line
                
                for col_index in range(col_num):
                    # TODO add sample std score here 
                    if abs(float(ori_str_scores[col_index]) - best_score[col_index]) < 1e-5:
                        one_best_line = one_best_line.replace(ori_str_scores[col_index], "\\best{" + ori_str_scores[col_index] + "}")
                best_lines.append(one_best_line)

            return best_lines


        method_order_list = [line.strip().replace('\\\\','\\') for line in read_txt(method_order_txt)]
        methodTag_info = read_json(all_info_json)
        caption = all_info_json.split("/")[-1].replace("_", "\_")

        main_table_lines = read_txt("./visualization/table_template/main_table_begin.txt")
        sup_table_part1_lines = read_txt("./visualization/table_template/sup1_table_begin.txt")
        sup_table_part2_lines = read_txt("./visualization/table_template/sup2_table_begin.txt")

        main_table_lines_sub = []
        sup_table_part1_lines_sub = []
        sup_table_part2_lines_sub = []

        main_table_lines_metrics_list = []
        sup_table_part1_lines_metrics_list = []
        sup_table_part2_lines_metrics_list = []

        for exp_index, method_tag in enumerate(method_order_list):
            
            one_latex_lines = [item for item in methodTag_info[method_tag][-1].items()] # TODO -1 is nyu normal 0 is m3d normal
            main_table_lines_sub.append(one_latex_lines[0][1] +"\\")
            main_table_lines_metrics_list.append(one_latex_lines[0][0])

            sup_table_part1_lines_sub.append(one_latex_lines[1][1] +"\\")
            sup_table_part1_lines_metrics_list.append(one_latex_lines[1][0])

            sup_table_part2_lines_sub.append(one_latex_lines[2][1] +"\\")
            sup_table_part2_lines_metrics_list.append(one_latex_lines[2][0])

            
            if ((exp_index + 1) in midrule_index) or (exp_index == len(method_order_list)-1):
                
                if "*" not in method_tag:
                    # import pdb;pdb.set_trace()
                    main_table_lines_sub = identify_best_for_subLines(main_table_lines_sub, main_table_lines_metrics_list)
                    sup_table_part1_lines_sub = identify_best_for_subLines(sup_table_part1_lines_sub, sup_table_part1_lines_metrics_list)
                    sup_table_part2_lines_sub = identify_best_for_subLines(sup_table_part2_lines_sub, sup_table_part2_lines_metrics_list)
                    
                main_table_lines += main_table_lines_sub
                sup_table_part1_lines += sup_table_part1_lines_sub
                sup_table_part2_lines += sup_table_part2_lines_sub
                if exp_index != len(method_order_list)-1:
                    main_table_lines.append("\midrule")
                    sup_table_part1_lines.append("\midrule")
                    sup_table_part2_lines.append("\midrule")

                main_table_lines_sub = []
                sup_table_part1_lines_sub = []
                sup_table_part2_lines_sub = []
                
        
        main_table_lines.append("\\bottomrule\end{tabular}} \caption{" + caption + "}\end{table}")
        sup_table_part1_lines.append("\\bottomrule\end{tabular}}")
        sup_table_part2_lines.append("\\bottomrule\end{tabular}}\caption{ Additional quantitative metrics for " + caption + "}\end{table*}")

        print(" ##################### main latex table ##################### ")
        for line in main_table_lines:
            print(line)

        print(" ##################### supplemental latex table ##################### ")
        for line in sup_table_part1_lines:
            print(line)
        
        for line in sup_table_part2_lines:
            print(line)


    def gen_latex_table_whole(self, method_order_txt, all_info_json, midrule_index):

        def identify_best_for_subLines(sublines, metrics_list):
            downmetrics = ['RMSE', 's-RMSE', 'Rel']
            upmetrics = ['SSIM','$d_{1.05}$','$d_{1.12}$','$d_{1.25}$','$d_{1.25^2','$d_{1.25^3}$']
            col_num = len(sublines[0].split("&")) - 3
            metrics_list = metrics_list[0].split(",")

            subline_score = []
            for one_line in sublines:
                subline_score.append([float(one_score.strip()) for one_score in one_line.replace('\\','').split("&")[3:]])
            subline_score = np.array(subline_score)
            
            # identify the best score in a column
            best_score = np.zeros(col_num)
            for col_index in range(col_num):
                metric = metrics_list[int(col_index/3)]
                if metric in downmetrics:
                    best_score[col_index] = subline_score[:,col_index].min()
                else:
                    best_score[col_index] = subline_score[:,col_index].max()
            
            # replace the best score with \best{}
            best_lines = []
            for one_line in sublines: 
                ori_str_scores = [one_score.strip() for one_score in one_line.replace('\\','').split("&")[3:]]
                one_best_line = one_line
                
                for col_index in range(col_num):
                    if abs(float(ori_str_scores[col_index]) - best_score[col_index]) < 1e-5:
                        one_best_line = one_best_line.replace(ori_str_scores[col_index], "\\best{" + ori_str_scores[col_index] + "}")
                best_lines.append(one_best_line)

            return best_lines


        method_order_list = [line.strip().replace('\\\\','\\') for line in read_txt(method_order_txt)]
        methodTag_info = read_json(all_info_json)
        caption = all_info_json.split("/")[-1].replace("_", "\_")

        main_table_lines = read_txt("./visualization/table_template/main_table_begin.txt")
        sup_table_part1_lines = read_txt("./visualization/table_template/sup1_table_begin.txt")
        sup_table_part2_lines = read_txt("./visualization/table_template/sup2_table_begin.txt")

        main_table_lines_sub = []
        sup_table_part1_lines_sub = []
        sup_table_part2_lines_sub = []

        main_table_lines_metrics_list = []
        sup_table_part1_lines_metrics_list = []
        sup_table_part2_lines_metrics_list = []

        for exp_index, method_tag in enumerate(method_order_list):
            
            one_latex_lines = [item for item in methodTag_info[method_tag][0].items()] # TODO -1 is nyu normal 0 is m3d normal
            main_table_lines_sub.append(one_latex_lines[0][1] +"\\")
            main_table_lines_metrics_list.append(one_latex_lines[0][0])

            sup_table_part1_lines_sub.append(one_latex_lines[1][1] +"\\")
            sup_table_part1_lines_metrics_list.append(one_latex_lines[1][0])

            sup_table_part2_lines_sub.append(one_latex_lines[2][1] +"\\")
            sup_table_part2_lines_metrics_list.append(one_latex_lines[2][0])

            
            if ((exp_index + 1) in midrule_index) or (exp_index == len(method_order_list)-1):
                
                if "*" not in method_tag:
                    # import pdb;pdb.set_trace()
                    main_table_lines_sub = identify_best_for_subLines(main_table_lines_sub, main_table_lines_metrics_list)
                    sup_table_part1_lines_sub = identify_best_for_subLines(sup_table_part1_lines_sub, sup_table_part1_lines_metrics_list)
                    sup_table_part2_lines_sub = identify_best_for_subLines(sup_table_part2_lines_sub, sup_table_part2_lines_metrics_list)
                    
                main_table_lines += main_table_lines_sub
                sup_table_part1_lines += sup_table_part1_lines_sub
                sup_table_part2_lines += sup_table_part2_lines_sub
                if exp_index != len(method_order_list)-1:
                    main_table_lines.append("\midrule")
                    sup_table_part1_lines.append("\midrule")
                    sup_table_part2_lines.append("\midrule")

                main_table_lines_sub = []
                sup_table_part1_lines_sub = []
                sup_table_part2_lines_sub = []
                
        
        main_table_lines.append("\\bottomrule\end{tabular}} \caption{" + caption + "}\end{table}")
        sup_table_part1_lines.append("\\bottomrule\end{tabular}}")
        sup_table_part2_lines.append("\\bottomrule\end{tabular}}\caption{ Additional quantitative metrics for " + caption + "}\end{table*}")

        print(" ##################### main latex table ##################### ")
        for line in main_table_lines:
            print(line)

        print(" ##################### supplemental latex table ##################### ")
        for line in sup_table_part1_lines:
            print(line)
        
        for line in sup_table_part2_lines:
            print(line)


    def get_std_score(self, args):
        from utils.mirror3d_metrics import Mirror3d_eval
        
        test_sample_name_list = []
        input_images = read_json(args.test_json)["images"]
        for one_info in input_images: 
            test_sample_name_list.append(one_info["hole_raw_path"].split("/")[-1])

        for item_index, item in enumerate(self.method_predFolder.items()):
            if args.multi_processing and item_index!=args.process_index:
                continue
            pred_folder = item[1]
            method_output_folder = os.path.split(pred_folder)[0]
            method_tag_long = item[0]
            Input_tag, refined_depth, method_tag = method_tag_long.split(",")
            if "ref" in refined_depth :
                refined_depth = True
            elif refined_depth == "raw" or refined_depth== "mesh" :
                refined_depth = False
            method_tag.replace('\\\\','\\')
            if "m3d" in self.dataset_main_folder:
                dataset_name = "m3d"
            elif "sacnnet" in self.dataset_main_folder:
                dataset_name = "sacnnet"
            else:
                dataset_name = "nyu"
            mirror3d_eval = Mirror3d_eval(refined_depth,logger=None, Input_tag=Input_tag, method_tag=method_tag, dataset=dataset_name)
            mirror3d_eval.set_cal_std(True)
            mirror3d_eval.set_min_threshold_filter(args.min_threshold_filter)
            mirror3d_eval.set_save_score_per_sample(True)
            for one_pred_name in tqdm(test_sample_name_list):
                if one_pred_name not in os.listdir(pred_folder):
                    continue
                one_pred_path = os.path.join(pred_folder, one_pred_name)
                if self.is_matterport3d:
                    depth_shift = 4000
                    color_image_path = os.path.join(self.dataset_main_folder, "raw", rreplace(one_pred_name, "d", "i"))
                else:
                    depth_shift = 1000
                    color_image_path = os.path.join(self.dataset_main_folder, "raw", one_pred_name)
                pred_depth = cv2.imread(one_pred_path, cv2.IMREAD_ANYDEPTH)


                mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth/depth_shift, depth_shift, color_image_path)
            mirror3d_eval.print_mirror3D_score()
            mirror3d_eval.save_sampleScore(method_output_folder=method_output_folder)
            

    def generate_paper_html(self, vis_saved_main_folder="", sample_num_per_page=50, template_path=""):
        """
        (1) under vis_saved_folder there should only be the vislization output sub-folders and nothing else

        Each line will show:
            1. Sample ID & method name 
            2. Color image
            3. Colored RMSE map
            4. Colored predict depth
            5. Front view
            6. Topdown view
        """

        if self.is_matterport3d:
            method_folder_list = ["GT","mesh-D","mesh-D+Mirror3DNet","saic-ref","saic-raw","saic+Mirror3DNet","BTS-ref","BTS-raw","BTS+Mirror3DNet","VNL-ref","VNL-raw","VNL+Mirror3DNet"]
        else:
            method_folder_list = ["GT","sensor-D","sensor-D+Mirror3DNet-m3d","saic-ref","saic-raw","saic+Mirror3DNet-m3d","BTS-ref","BTS-raw","BTS+Mirror3DNet-m3d","VNL-ref","VNL-raw","VNL+Mirror3DNet-m3d"]
        # TODO if want to sort by RMSE can sort the self.color_img_list by RMSE score here 
        colorImgSubset_list = [self.color_img_list[x:x+sample_num_per_page] for x in range(0, len(self.color_img_list), sample_num_per_page)]
        for html_index, one_colorSubset in enumerate(colorImgSubset_list):
            
            with open(template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")
            for one_color_img_path in one_colorSubset:
                sample_name = os.path.split(one_color_img_path)[-1]
                sample_id = sample_name.split(".")[0]
                if self.is_matterport3d:
                    one_depth_sample_name = rreplace(sample_name, "i", "d")
                else:
                    one_depth_sample_name = sample_name
                
                new_table = soup.new_tag("table")
                new_table["style"] = "width: 100%%; margin-left: auto; margin-right: auto;"
                soup.body.div.append(new_table)
                for method_index, one_method_name in enumerate(method_folder_list):
                    row_num = int(method_index / 3)
                    one_method_folder_path = os.path.join(vis_saved_main_folder, one_method_name)
                    one_RMSE_map = os.path.join(one_method_folder_path, "colored_pred_error_map", one_depth_sample_name)
                    one_predD_map = os.path.join(one_method_folder_path, "colored_pred_depth", one_depth_sample_name)
                    one_front_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "front_{}".format(sample_name))
                    one_topdown_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "topdown_{}".format(sample_name))

                    if method_index % 3 == 0 and method_index != (len(method_folder_list) - 1):
                        is_first_col = True

                    if row_num == 0 and is_first_col:
                        heading = soup.new_tag("tr")
                        new_table.append(heading)
                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        heading.append(one_blank)

                        heading["class"] = "begin-text"
                        one_blank = soup.new_tag("td")
                        heading.append(one_blank)


                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        text.string = " Ground Truth "
                        text["style"] = "text-align: center; margin-bottom:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        if self.is_matterport3d:
                            text.string = "MP3D-mesh rendered depth"
                        else:
                            text.string = "Raw Sensor"
                        text["style"] = "text-align: center; margin-bottom:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        text.string = "Pred-Refined"
                        text["style"] = "text-align: center; margin-bottom:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                    if method_index % 3 == 0:
                        new_tr = soup.new_tag("tr")
                        new_table.append(new_tr)


                        

                    if method_folder_list[method_index] == "saic-ref":
                        pass
                    if row_num == 1 and is_first_col:
                        # add mesh-D for matterport3d; sensor-D for NYUv2 and Scannet
                        one_color_img = soup.new_tag("td")
                        one_color_img["class"] = "one-item"
                        raw_depth_map = os.path.relpath(os.path.join(os.path.join(vis_saved_main_folder, method_folder_list[1]), "colored_pred_depth", one_depth_sample_name), self.output_folder)
                        one_color_img.append(soup.new_tag('img', src=raw_depth_map))
                        new_tr.append(one_color_img)
                        is_first_col = False
                        # add info
                        text = soup.new_tag("p")
                        text.string = "saic"
                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-method"
                        one_blank.append(text)
                        new_tr.append(one_blank)
                    elif row_num == 2 and is_first_col:
                        # add color image
                        one_color_img = soup.new_tag("td")
                        one_color_img["class"] = "one-item"
                        color_img_path = os.path.relpath(one_color_img_path, self.output_folder)
                        one_color_img.append(soup.new_tag('img', src=color_img_path))
                        new_tr.append(one_color_img)
                        # add info 
                        text = soup.new_tag("p")
                        text.string = "bts"
                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-method"
                        one_blank.append(text)
                        new_tr.append(one_blank)
                        is_first_col = False
                    elif row_num == 0 and is_first_col :
                        text = soup.new_tag("p")
                        text.string = "Input depth(D)"
                        text["style"] = " margin : 0;text-align: center;"
                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-bottom" 
                        one_blank.append(text)
                        new_tr.append(one_blank)
                        # add info
                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-text"
                        new_tr.append(one_blank)
                        is_first_col = False
                    elif row_num == 3 and is_first_col :

                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-top"

                        text = soup.new_tag("p")
                        text.string = "color(RGB)"
                        text["style"] = " margin : 0;text-align: center;"
                        one_blank.append(text)

                        text = soup.new_tag("p")
                        text.string = sample_name
                        text["style"] = "text-align: center;"
                        one_blank.append(text)

                        new_tr.append(one_blank)


                        # add info
                        text = soup.new_tag("p")
                        text.string = "vnl"
                        one_blank = soup.new_tag("td")
                        one_blank["class"] = "begin-method"
                        one_blank.append(text)
                        new_tr.append(one_blank)
                        is_first_col = False

                    # Append RMSE map to one line in HTML
                    one_color_img = soup.new_tag("td")
                    one_color_img["class"] = "one-item"
                    one_RMSE_map = os.path.relpath(one_RMSE_map, self.output_folder)
                    one_color_img.append(soup.new_tag('img', src=one_RMSE_map))
                    new_tr.append(one_color_img)

                    # Append predict depth to one line in HTML
                    one_color_img = soup.new_tag("td")
                    one_color_img["class"] = "one-item"
                    one_predD_map = os.path.relpath(one_predD_map, self.output_folder)
                    one_color_img.append(soup.new_tag('img', src=one_predD_map))
                    new_tr.append(one_color_img)

                    # Append one_topdown_view_img image to one line in HTML
                    one_color_img = soup.new_tag("td")
                    one_color_img["class"] = "one-item"
                    one_topdown_view_img = os.path.relpath(one_topdown_view_img, self.output_folder)
                    topdown_img = soup.new_tag('img', src=one_topdown_view_img)
                    topdown_img["style"] = "max-height: 220px; width:100%;" 
                    one_color_img.append(topdown_img)
                    new_tr.append(one_color_img)

                    if row_num == 3 and method_index == (len(method_folder_list) - 1):
                        heading = soup.new_tag("tr")
                        new_table.append(heading)
                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        heading.append(one_blank)

                        heading["class"] = "begin-text"
                        one_blank = soup.new_tag("td")
                        heading.append(one_blank)


                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        if self.is_matterport3d:
                            text.string = "MP3D-mesh-ref "
                        else:
                            text.string = "NYUv2-ref "
                        text["style"] = "text-align: center; margin-top:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        if self.is_matterport3d:
                            text.string = "MP3D-mesh-raw "
                        else:
                            text.string = "MP3D-mesh"
                        text["style"] = "text-align: center; margin-top:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                        heading["class"] = "one-item"
                        one_blank = soup.new_tag("td")
                        one_blank["colspan"] = "3" 
                        text = soup.new_tag("p")
                        text.string = "+Mirror3DNet"
                        text["style"] = "text-align: center; margin-top:0"
                        one_blank.append(text)                       
                        heading.append(one_blank)

                
            html_path = os.path.join(self.output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)
            print("result visulisation saved in link {}".format(html_path.replace("/project/3dlg-hcvc/mirrors/www","http://aspis.cmpt.sfu.ca/projects/mirrors")))
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="6")
    parser.add_argument(
        '--test_json', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json")
    parser.add_argument(
        '--method_predFolder_txt', default="/project/3dlg-hcvc/mirrors/www/notes/nyu_vis_0418.txt")
    parser.add_argument(
        '--dataset_main_folder', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu")
    parser.add_argument(
        '--process_index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    parser.add_argument('--overwrite', help='overwrite files under --output_folder or not',action='store_true')
    parser.add_argument(
        '--f', default=537, type=int, help="camera focal length")
    parser.add_argument(
        '--pred_w', default=640, type=int, help="width of the visilization window")
    parser.add_argument(
        '--pred_h', default=512, type=int, help="width of the visilization window")
    parser.add_argument(
        '--window_w', default=800, type=int, help="width of the visilization window")
    parser.add_argument(
        '--window_h', default=800, type=int, help="height of the visilization window")
    parser.add_argument(
        '--sample_num_per_page', default=100, type=int, help="height of the visilization window")
    parser.add_argument(
        '--vis_saved_folder', default="/project/3dlg-hcvc/mirrors/www/cr_vis/nyu_result_vis")
    parser.add_argument(
        '--output_folder', default="/project/3dlg-hcvc/mirrors/www/cr_vis/nyu_html")
    parser.add_argument(
        '--method_folder_list', nargs='+', default="", type=str)
    parser.add_argument("--midrule_index", nargs="+", type=int, default=[2,5], help="add /midrule in after these liens; index start from 1") 
    parser.add_argument(
        '--template_path', default="visualization/result_vis_template.html", type=str)
    parser.add_argument(
        '--view_mode', default="topdown", help="object view angle : (1) topdown (2) front")
    parser.add_argument(
        '--method_order_txt', default="", type=str)
    parser.add_argument(
        '--all_info_json', default="output/ref_m3d_result.json", type=str)
    parser.add_argument('--min_threshold_filter', help='',action='store_true')
    parser.add_argument('--add_rmse', help='add rmse to result visulization or not',action='store_true')
    args = parser.parse_args()

    vis_tool = Dataset_visulization(pred_w = args.pred_w, pred_h = args.pred_h, dataset_main_folder=args.dataset_main_folder, process_index=args.process_index, \
                                    multi_processing=args.multi_processing, f=args.f, test_json=args.test_json, \
                                    output_folder=args.output_folder, overwrite=args.overwrite, \
                                    window_w=args.window_w, window_h=args.window_h, view_mode=args.view_mode)
    if args.stage == "1":
        # vis_tool.generate_method_pred(args.method_predFolder_txt)
        # vis_tool.generate_pcd_for_whole_dataset()
        vis_tool.generate_method_pred(args.method_predFolder_txt)
        vis_tool.generate_color_depth_for_all_pred()
    elif args.stage == "2":
        vis_tool.generate_html(vis_saved_main_folder=args.vis_saved_folder, sample_num_per_page=args.sample_num_per_page, template_path=args.template_path)
    elif args.stage == "3":
        vis_tool.gen_latex_table_sep(args.method_predFolder_txt, args.midrule_index)
    elif args.stage == "4":
        vis_tool.gen_latex_table_whole(args.method_order_txt, args.all_info_json, args.midrule_index)
    elif args.stage == "5":
        vis_tool.generate_method_pred(args.method_predFolder_txt)
        vis_tool.get_std_score(args)
    elif args.stage == "6":
        vis_tool.generate_paper_html(vis_saved_main_folder=args.vis_saved_folder, sample_num_per_page=args.sample_num_per_page, template_path="visualization/paper_vis_template.html")