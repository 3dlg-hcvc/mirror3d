
import argparse
from utils.general_utlis import *
import shutil
import os
import bs4
import tqdm

class Verification():

    def __init__(self, data_main_folder="", video_main_folder="", output_folder="", video_num_per_page=20, template_path = "./annotation/plane_annotation/template.html", show_mesh_depth=True):
        """
        Initilization
        """
        self.data_main_folder = data_main_folder
        if "m3d" not in self.data_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.video_main_folder = video_main_folder
        self.output_folder = output_folder
        self.video_num_per_page = video_num_per_page
        self.template_path = template_path
        os.makedirs(self.output_folder, exist_ok=True)
        self.show_mesh_depth = show_mesh_depth


    def sort_data_to_reannotate(self, error_id_path, waste_id_path, move):
        """
        Make a copy of error sample

        Args:
            self.video_main_folder : Dataset main folder.
            self.output_folder : Path to save the copy.
            error_id_path : .txt file that contains the error sample's id.
            waste_id_path : .txt file path that contains invalid sample's id.
        """
        if os.path.exists(waste_id_path):
            invalid_id_list = set(read_txt(waste_id_path))
        else:
            invalid_id_list = []
        if os.path.exists(error_id_path):
            error_id_list = set(read_txt(error_id_path))
        else:
            error_id_list = []
        os.makedirs(self.output_folder, exist_ok=True)
        print("A copy of error data is saved to : {}".format(self.output_folder))
        
        re_anno_id_list = set(error_id_list) - set(invalid_id_list)

        # move invalid sample to "only_mask" folder
        color_image_folder = os.path.join(self.data_main_folder, "mirror_color_images")
        color_name_list = os.listdir(color_image_folder)
        color_name_list.sort()
        for one_color_img_name in color_name_list:
            color_img_path = os.path.join(color_image_folder, one_color_img_name)
            sample_id = color_img_path.split("/")[-1].split(".")[0]
            if sample_id in invalid_id_list:
                if self.is_matterport3d:
                    depth_sample_id = "{}_{}_{}".format(sample_id.split("_")[0], sample_id.split("_")[1].replace("i", "d"), sample_id.split("_")[2])
                    command = "find -L {} -type f | grep {}".format(self.data_main_folder, depth_sample_id)
                    for src_path in os.popen(command).readlines():
                        src_path = src_path.strip()
                        dst_folder = os.path.split(dst_path)[0]
                        if os.path.exists(dst_path) and move:
                            continue
                        os.makedirs(dst_folder, exist_ok=True)
                        if move:
                            print("moving {} to {}".format(src_path, dst_folder))
                            shutil.move(src_path, dst_folder)
                        else:
                            print("copying {} to {}".format(src_path, dst_folder))
                            shutil.copy(src_path, dst_path)

                command = "find -L {} -type f | grep {}".format(self.data_main_folder, sample_id)
                for src_path in os.popen(command).readlines():
                    src_path = src_path.strip()
                    dst_path = os.path.join(self.output_folder, src_path.split("/")[-2], src_path.split("/")[-1])
                    dst_folder = os.path.split(dst_path)[0]
                    if os.path.exists(dst_path) and move:
                        continue
                    os.makedirs(dst_folder, exist_ok=True)
                    if move:
                        print("moving {} to {}".format(src_path, dst_folder))
                        shutil.move(src_path, dst_folder)
                    else:
                        print("copying {} to {}".format(src_path, dst_folder))
                        shutil.copy(src_path, dst_path)

                
            elif sample_id in re_anno_id_list:
                if self.is_matterport3d:
                    depth_sample_id = "{}_{}_{}".format(sample_id.split("_")[0], sample_id.split("_")[1].replace("i", "d"), sample_id.split("_")[2])
                    command = "find -L {} -type f | grep {}".format(self.data_main_folder, depth_sample_id)
                    for src_path in os.popen(command).readlines():
                        src_path = src_path.strip()
                        dst_path = src_path.replace(self.data_main_folder, self.output_folder)
                        dst_folder = os.path.split(dst_path)[0]
                        if os.path.exists(dst_path) and move:
                            continue
                        os.makedirs(dst_folder, exist_ok=True)
                        if move:
                            print("moving {} to new_folder {}".format(src_path, dst_folder))
                            shutil.move(src_path, dst_folder)
                        else:
                            print("copying {} to new_folder {}".format(src_path, dst_folder))
                            shutil.copy(src_path, dst_path)

                command = "find -L {} -type f | grep {}".format(self.data_main_folder, sample_id)
                for src_path in os.popen(command).readlines():
                    src_path = src_path.strip()
                    dst_path = src_path.replace(self.data_main_folder, self.output_folder)
                    dst_folder = os.path.split(dst_path)[0]
                    if os.path.exists(dst_path) and move:
                        continue
                    os.makedirs(dst_folder, exist_ok=True)
                    if move:
                        print("moving {} to new_folder {}".format(src_path, dst_folder))
                        shutil.move(src_path, dst_folder)
                    else:
                        print("copying {} to new_folder {}".format(src_path, dst_folder))
                        shutil.copy(src_path, dst_path)

                
    def generate_html(self):
        """
        Generate html to show video; all views for one sample is shown in one line;
        """

        # Get video folder name (e.g. video_front; video_topdown)
        video_folder_list = []
        for item in os.listdir(self.video_main_folder):
            if item.count('video', 0, len(item)) > 0 :
                video_folder_list.append(item)

        video_folder_list = ['video_topdown','video_front']
        one_video_folder_name = video_folder_list[0]
        one_video_folder_path = os.path.join(self.video_main_folder, one_video_folder_name)
        video_path_list = os.listdir(one_video_folder_path)
        video_path_list.sort()
        videoSubset_list = [video_path_list[x:x+self.video_num_per_page] for x in range(0, len(video_path_list), self.video_num_per_page)]
        for html_index, one_videoSubset in enumerate(videoSubset_list):
            
            with open(self.template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")

            for video_index ,one_video_name in enumerate(one_videoSubset):
            # Get video path for one instance (all put in one line)
                if self.is_matterport3d:
                    one_line_video = []
                    for i in video_folder_list:
                        one_path = os.path.join(self.video_main_folder, i, one_video_name)
                        if one_path.find("mesh") > 0 and self.is_matterport3d:
                            one_line_video.append(one_path)
                        if one_path.find("sensor") > 0 and not self.show_mesh_depth:
                            one_line_video.append(one_path)

                else: 
                    one_line_video = [os.path.join(self.video_main_folder, i, one_video_name) for i in video_folder_list]

                sample_color_img_name = "{}.png".format(one_video_name.split("_idx_")[0])

                new_div = soup.new_tag("div")
                new_div['class'] = "one-instance"
                
                soup.body.append(new_div)


                # Append text to one line in HTML
                one_text = soup.new_tag("div")
                one_text["class"] = "one-item"
                if not self.is_matterport3d:
                    one_text['style'] = "padding-top: 100px;font-size: 25pt;"
                else:
                    one_text['style'] = "padding-top: 100px;"

                one_text.string = sample_color_img_name.split(".")[0]
                new_div.append(one_text)

                # Append color image to one line in HTML
                one_color_img = soup.new_tag("div")
                one_video_path = one_line_video[0]
                color_img = soup.new_tag("div")
                color_img["class"] = "one-item"
                color_img_path = os.path.relpath(os.path.join(self.data_main_folder, "mirror_color_images", "{}.jpg".format(one_video_path.split("/")[-1].split("_idx_")[0])), self.output_folder)
                color_img.append(soup.new_tag('img', src=color_img_path))
                one_color_img.append(color_img)
                new_div.append(one_color_img)

                # Append colored dpeth image to one line in HTML
                one_colored_depth = soup.new_tag("div")
                colored_depth = soup.new_tag("div")
                colored_depth["class"] = "one-item"
                if self.is_matterport3d:
                    sample_name = rreplace(color_img_path.split("/")[-1], "i", "d").replace(".jpg",".png")
                    if self.show_mesh_depth:
                        colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "refined_mesh_colored_depth", sample_name),self.output_folder)
                    else:
                        colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "refined_sensorD_colored_depth", sample_name),self.output_folder)
                else:
                    sample_name = color_img_path.split("/")[-1].replace(".jpg",".png")
                    colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "refined_sensorD_colored_depth", sample_name),self.output_folder)
                colored_depth.append(soup.new_tag('img', src=colored_depth_path))
                one_colored_depth.append(colored_depth)
                new_div.append(one_colored_depth)

                # Append on video div to one_line_video div
                for one_video_path in one_line_video:

                    one_video_div = soup.new_tag("div")
                    one_video_div["class"] = "one-item"
                    new_div.append(one_video_div)

                    front_video = soup.new_tag("video")
                    front_video["class"] = "lazy-video"
                    front_video["controls"] = "True"
                    front_video["autoplay"] = "True"
                    front_video["muted"] = "True"
                    front_video["loop"] = "True"
                    
                    new_link = soup.new_tag("source")
                    new_link["data-src"] = os.path.relpath(one_video_path, self.output_folder)
                    new_link["type"] = "video/mp4"
                    front_video.append(new_link)
                    one_video_div.append(front_video)
                    new_div.append(one_video_div)

                
                
            
            html_path = os.path.join(self.output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)
            
            print("{} videos saved in {} link {}".format(video_index, html_path, html_path.replace("/project/3dlg-hcvc/mirrors/www","http://aspis.cmpt.sfu.ca/projects/mirrors")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument('--move', help='For step 2 move the smaple to output_folder or copy the sample to output_folder',action='store_true')
    parser.add_argument('--show_mesh_depth', help='for Matterport3d dataset, only visulize mesh depth or not',action='store_true')
    parser.add_argument(
        '--video_main_folder', default="", help="dataset main folder / video main folder (under which have video_front/ video_topdown folders)")
    parser.add_argument(
        '--data_main_folder', default="", help="dataset main folder / video main folder (under which have raw instance_mask ...)")
    parser.add_argument(
        '--error_list', default="")
    parser.add_argument(
        '--waste_list', default="")
    parser.add_argument(
        '--output_folder', default="")
    parser.add_argument('--video_num_per_page', default=150, type=int)
    args = parser.parse_args()

    if args.stage == "1":
        verify = Verification(data_main_folder=args.data_main_folder, video_main_folder=args.video_main_folder, output_folder=args.output_folder,video_num_per_page =args.video_num_per_page, show_mesh_depth=args.show_mesh_depth)
        verify.generate_html()
    elif args.stage == "2":
        verify = Verification(data_main_folder=args.data_main_folder, video_main_folder=args.video_main_folder, output_folder=args.output_folder, video_num_per_page =args.video_num_per_page, show_mesh_depth=args.show_mesh_depth)
        verify.sort_data_to_reannotate(args.error_list, args.waste_list, args.move)
