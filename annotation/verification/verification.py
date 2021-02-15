
import argparse
from utils.general_utlis import *
import shutil
import os
import bs4

class Verification():

    def __init__(self, data_main_folder, output_folder, video_num_per_page=20, template_path = "./annotation/verification/template.html"):
        """
        Initilization
        """
        self.data_main_folder = data_main_folder
        self.output_folder = output_folder
        self.video_num_per_page = video_num_per_page
        self.template_path = template_path
        os.makedirs(self.output_folder, exist_ok=True)


    def sort_data_to_reannotate(self, error_list):
        """
        Make a copy of error sample

        Args:
            self.data_main_folder : Dataset main folder.
            self.output_folder : Path to save the copy.
            error_list : .txt file that contains the error sample's name.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        print("A copy of error data is saved to : {}".format(self.output_folder))
        for one_raw_name in read_txt(error_list):
            color_img_path = os.path.join(self.data_main_folder, "raw", one_raw_name)
            mask_path = os.path.join(self.data_main_folder, "instance_mask", one_raw_name)
            img_info_path = os.path.join(self.data_main_folder, "img_info", one_raw_name.replace(".png", ".json"))
            if self.data_main_folder.find("m3d") > 0:
                raw_depth_path = os.path.join(self.data_main_folder, "hole_raw_depth", rreplace(one_raw_name, "i", "d"))
                refined_depth_path = os.path.join(self.data_main_folder, "hole_refined_depth", rreplace(one_raw_name, "i", "d"))
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
        # Embed one video
        def video_embed(soup, one_video_div, view_link):
            front_video = soup.new_tag("video")
            front_video["class"] = "lazy-video"
            front_video["controls"] = "True"
            front_video["autoplay"] = "True"
            front_video["muted"] = "True"
            front_video["loop"] = "True"
            front_video["src"] = ""
            one_video_div.append(front_video)
            new_link = soup.new_tag("source")
            new_link["data-src"] = view_link
            new_link["type"] = "video/mp4"
            front_video.append(new_link)

        # Get video folder name (e.g. video_front; video_topdown)
        video_folder_list = []
        for item in os.listdir(self.data_main_folder):
            if item.count('video', 0, len(item)) > 0 :
                video_folder_list.append(item)

        one_video_folder_name = video_folder_list[0]
        one_video_folder_path = os.path.join(self.data_main_folder, one_video_folder_name)
        video_path_list = os.listdir(one_video_folder_path)
        videoSubset_list = [video_path_list[x:x+self.video_num_per_page] for x in range(0, len(video_path_list), self.video_num_per_page)]
        for html_index, one_videoSubset in enumerate(videoSubset_list):

            with open(self.template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")

            for one_video_name in one_videoSubset:
            # Get video path for one instance (all put in one line)
                one_line_video = [os.path.join(self.data_main_folder, i, one_video_name) for i in video_folder_list]
                sample_color_img_name = "{}.png".format(one_video_name.split("_idx_")[0])

                new_div = soup.new_tag("div")
                new_div['id'] = "one_instance_video"
                soup.body.append(new_div)

                new_sub_text_div = soup.new_tag("div")
                new_sub_text_div["class"] = "text"
                new_div.append(new_sub_text_div)

                new_text = soup.new_tag("b")
                new_text.string = sample_color_img_name
                new_sub_text_div.append(new_text)

                # Append on video div to one_line_video div
                for one_video_path in one_line_video:
                    one_video_div = soup.new_tag("div")
                    one_video_div["class"] = "video"
                    new_div.append(one_video_div)

                    front_video = soup.new_tag("video")
                    front_video["class"] = "lazy-video"
                    front_video["controls"] = "True"
                    front_video["autoplay"] = "True"
                    front_video["muted"] = "True"
                    front_video["loop"] = "True"
                    front_video["src"] = ""
                    
                    new_link = soup.new_tag("source")
                    new_link["data-src"] = one_video_path
                    new_link["type"] = "video/mp4"
                    front_video.append(new_link)
                    one_video_div.append(front_video)
            
            html_path = os.path.join(self.output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="1")
    parser.add_argument(
        '--data_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test", help="dataset main folder / video main folder")
    parser.add_argument(
        '--error_list', default="")
    parser.add_argument(
        '--output_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test/html")
    parser.add_argument('--video_num_per_page', default=20, type=int)
    args = parser.parse_args()

    if args.stage == "1":
        args.data_main_folder = "/Users/tanjiaqi/Desktop/SFU/mirror3D/test/hole_refined_ply" # TODO delete later
        verify = Verification(args.data_main_folder, args.output_folder)
        verify.generate_html()
    elif args.stage == "2":
        verify = Verification(args.data_main_folder, args.output_folder)
        verify.sort_data_to_reannotate(args.error_list)
