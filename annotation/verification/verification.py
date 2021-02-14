
import argparse
from utils.general_utlis import *
import shutil


def sort_data_to_reannotate(error_list, data_main_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print("A copy of error data is saved to : {}".format(output_folder))

    for one_raw_name in read_txt(error_list):
        color_img_path = os.path.join(data_main_folder, "raw", one_raw_name)
        mask_path = os.path.join(data_main_folder, "instance_mask", one_raw_name)
        img_info_path = os.path.join(data_main_folder, "img_info", one_raw_name.replace(".png", ".json"))
        if data_main_folder.find("m3d") > 0:
            raw_depth_path = os.path.join(data_main_folder, "hole_raw_depth", rreplace(one_raw_name, "i", "d"))
            refined_depth_path = os.path.join(data_main_folder, "hole_refined_depth", rreplace(one_raw_name, "i", "d"))
            img_to_copy = [color_img_path, mask_path, img_info_path, raw_depth_path, refined_depth_path]
            for src_path in img_to_copy:
                src_type = src_path.split("/")[-2]
                dst_path_folder = os.path.join(output_folder, src_type)
                os.makedirs(dst_path_folder)
                dst_path = os.path.join(dst_path_folder, src_path.split("/")[-1])
                shutil.copy(src_path, dst_path)
                print("file copy to {}".format(dst_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--data_main_folder', default="/Users/tanjiaqi/Desktop/SFU/mirror3D/test")
    parser.add_argument(
        '--error_list', default="")
    parser.add_argument(
        '--output_folder', default="")

    args = parser.parse_args()
    sort_data_to_reannotate(args.error_list, args.data_main_folder, args.output_folder)
