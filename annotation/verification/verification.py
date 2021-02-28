
import argparse
from utils.general_utlis import *
import shutil
import os
import bs4

class Verification():

    def __init__(self, data_main_folder="", video_main_folder="", output_folder="", video_num_per_page=20, template_path = "./annotation/verification/template.html", show_mesh_depth=True):
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


    def sort_data_to_reannotate(self, error_list):
        """
        Make a copy of error sample

        Args:
            self.video_main_folder : Dataset main folder.
            self.output_folder : Path to save the copy.
            error_list : .txt file that contains the error sample's name.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        print("A copy of error data is saved to : {}".format(self.output_folder))
        for one_raw_id in read_txt(error_list):
            one_raw_name = "{}.png".format(one_raw_id)
            color_img_path = os.path.join(self.data_main_folder, "raw", one_raw_name)
            mask_path = os.path.join(self.data_main_folder, "instance_mask", one_raw_name)
            img_info_path = os.path.join(self.data_main_folder, "img_info", one_raw_name.replace(".png", ".json"))
            if self.data_main_folder.find("m3d") > 0:
                raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth", rreplace(one_raw_name, "i", "d"))
                refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth", rreplace(one_raw_name, "i", "d"))
            else:
                raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth",one_raw_name)
                refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth",one_raw_name)
            img_to_copy = [color_img_path, mask_path, img_info_path, raw_depth_path, refined_depth_path]
            for src_path in img_to_copy:
                src_type = src_path.split("/")[-2]
                dst_path_folder = os.path.join(self.output_folder, src_type)
                os.makedirs(dst_path_folder)
                dst_path = os.path.join(dst_path_folder, src_path.split("/")[-1])
                shutil.copy(src_path, dst_path)
                print("file copy to {}".format(dst_path))

    def generate_html(self):
        """
        Generate html to show video; all views for one sample is shown in one line;
        """

        # Get video folder name (e.g. video_front; video_topdown)
        video_folder_list = []
        for item in os.listdir(self.video_main_folder):
            if item.count('video', 0, len(item)) > 0 :
                video_folder_list.append(item)

        one_video_folder_name = video_folder_list[0]
        one_video_folder_path = os.path.join(self.video_main_folder, one_video_folder_name)
        video_path_list = os.listdir(one_video_folder_path)
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
                        if one_path.find("hole") > 0 and not self.show_mesh_depth:
                            one_line_video.append(one_path)

                else: 
                    one_line_video = [os.path.join(self.video_main_folder, i, one_video_name) for i in video_folder_list]

                sample_color_img_name = "{}.png".format(one_video_name.split("_idx_")[0])

                new_div = soup.new_tag("div")
                new_div['class'] = "one-instance"
                soup.body.append(new_div)


                # Append text to one line in HTML
                one_text = soup.new_tag("div")
                text_div = soup.new_tag("div")
                text_div["class"] = "text-div"
                new_text = soup.new_tag("p")
                new_text["class"] = "one-text"
                new_text.string = sample_color_img_name.split(".")[0]
                text_div.append(new_text)
                one_text.append(text_div)
                new_div.append(one_text)

                # Append color image to one line in HTML
                one_color_img = soup.new_tag("div")
                one_video_path = one_line_video[0]
                color_img = soup.new_tag("div")
                color_img["class"] = "one-item"
                color_img_path = os.path.relpath(os.path.join(self.data_main_folder, "raw", "{}.png".format(one_video_path.split("/")[-1].split("_idx_")[0])), self.output_folder)
                color_img.append(soup.new_tag('img', src=color_img_path))
                one_color_img.append(color_img)
                new_div.append(one_color_img)

                # Append colored dpeth image to one line in HTML
                one_colored_depth = soup.new_tag("div")
                colored_depth = soup.new_tag("div")
                colored_depth["class"] = "one-item"
                if self.is_matterport3d:
                    sample_name = rreplace(color_img_path.split("/")[-1], "i", "d")
                    if self.show_mesh_depth:
                        colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "mesh_refined_colored_depth", sample_name),self.output_folder)
                    else:
                        colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "hole_refined_colored_depth", sample_name),self.output_folder)
                else:
                    sample_name = color_img_path.split("/")[-1]
                    colored_depth_path = os.path.relpath(os.path.join(self.video_main_folder,  "hole_refined_colored_depth", sample_name),self.output_folder)
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
            print("{} videos saved in {}".format(video_index, html_path))
            exit() # TODO delete

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument('--show_mesh_depth', help='for Matterport3d dataset, only visulize mesh depth or not',action='store_true')
    parser.add_argument(
        '--video_main_folder', default="/project/3dlg-hcvc/mirrors/www/final_verification/nyu/hole_refined_ply", help="dataset main folder / video main folder (under which have video_front/ video_topdown folders)")
    parser.add_argument(
        '--data_main_folder', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise", help="dataset main folder / video main folder")
    parser.add_argument(
        '--error_list', default="/project/3dlg-hcvc/mirrors/www/final_verification/nyu/error.txt")
    parser.add_argument(
        '--output_folder', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/dataset_vis/nyu")
    parser.add_argument('--video_num_per_page', default=150, type=int)
    args = parser.parse_args()

    if args.stage == "1":
        verify = Verification(data_main_folder=args.data_main_folder, video_main_folder=args.video_main_folder, output_folder=args.output_folder,video_num_per_page =args.video_num_per_page, show_mesh_depth=args.show_mesh_depth)
        verify.generate_html()
    elif args.stage == "2":
        verify = Verification(data_main_folder=args.data_main_folder, video_main_folder=args.video_main_folder, output_folder=args.output_folder, video_num_per_page =args.video_num_per_page, show_mesh_depth=args.show_mesh_depth)
        verify.sort_data_to_reannotate(args.error_list)
