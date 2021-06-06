import argparse
import os
from mirror3d.utils.general_utils import *
from tqdm import tqdm


def generate_symlinks(unzipped_folder_path):
    if "nyu" in unzipped_folder_path:
        symlink_info = read_txt("mirror3d/dataset/symlink_infer/nyu_symlink.txt")
    elif "mp3d" in unzipped_folder_path:
        symlink_info = read_txt("mirror3d/dataset/symlink_infer/mp3d_symlink.txt")
    elif "scannet" in unzipped_folder_path:
        symlink_info = read_txt("mirror3d/dataset/symlink_infer/scannet_symlink.txt")
    else:
        print("Can't find nyu/ mp3d/ scannet in the unzip folder path, please input a valid --unzipped_folder_path")

    for item in tqdm(symlink_info):
        src_path, to_link_path = item.split()
        src_path = os.path.join(unzipped_folder_path, src_path)
        to_link_path = os.path.join(unzipped_folder_path, to_link_path)
        to_link_folder = os.path.split(to_link_path)[0]
        os.makedirs(to_link_folder, exist_ok=True)
        if not os.path.exists(src_path):
            print("source path {} not exists!".format(src_path))
            continue
        command = "ln -s {} {}".format(os.path.abspath(src_path), os.path.abspath(to_link_path))
        os.system(command)
    print("Symlink generation finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--unzipped_folder_path', default="")
    args = parser.parse_args()
    generate_symlinks(args.unzipped_folder_path)
