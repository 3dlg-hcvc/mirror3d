import argparse
import os
from utils.general_utlis import *


def generate_symlinks(unzip_folder_path):

    if "nyu" in unzip_folder_path:
        symlink_info = read_txt("metadata/nyu_symlink.txt")
    elif "mp3d" in unzip_folder_path:
        symlink_info = read_txt("metadata/mp3d_symlink.txt")
    elif "scannet" in unzip_folder_path:
        symlink_info = read_txt("metadata/scannet_symlink.txt")
    else:
        print("Can't find nyu/ mp3d/ scannet in the unzip folder path, please input a valid --unzip_folder_path")
    
    for item in symlink_info:
        src_path, to_link_path = item.split() 
        src_path = os.path.join(unzip_folder_path, src_path)
        to_link_path = os.path.join(unzip_folder_path, to_link_path)
        to_link_folder = os.path.split(to_link_path)[0]
        os.makedirs(to_link_folder, exist_ok=True)
        if not os.path.exists(src_path):
            print("source path {} not exists!".format(src_path))
            continue
        command = "ln -s {} {}".format(src_path, to_link_path)
        os.system(command)
    print("Finished generating symlinks!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--unzip_folder_path', default="")
    args = parser.parse_args()
    generate_symlinks(args.unzip_folder_path)
