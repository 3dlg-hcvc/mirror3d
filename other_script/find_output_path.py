import os
import numpy as np
import cv2
import argparse
from utils.general_utlis import *


def find_line(txt_list_path, dataset):
    while 1:
        found = False
        print("###########################################################################################")
        line_to_search = input("please input the lines to search :")
        if line_to_search == "q":
            exit()
        filename_lines = dict()
        txt_list = read_txt(txt_list_path)
        for one_txt_path in txt_list:
            one_lines = read_txt(one_txt_path)
            filename_lines[one_txt_path] = one_lines
        
        for one_item in  filename_lines.items():
            for one_line in one_item[1]:
                if one_line.find(line_to_search) > 0:
                    log_path = one_item[0]
                    output_folder = os.path.split(log_path)[0]
                    command = "find {} -type f | grep png".format(output_folder)
                    png_paths = [line for line in  os.popen(command).readlines()]
                    if dataset == "m3d":
                        if len(png_paths) < 650:
                            print(" find in  {}".format(log_path))
                            print(output_folder, "no png")
                            continue
                    else:
                        if len(png_paths) < 50:
                            print(" find in  {}".format(log_path))
                            print(output_folder, "no png")
                            continue
                    print(" find in  {}".format(log_path))
                    found = True
        if not found:
            print("found nothing :(")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')

    parser.add_argument(
        '--txt_list_path', default="/local-scratch/jiaqit/exp/Mirror3D/cr_output/eval_log_sum.txt")
    parser.add_argument(
        '--dataset', default="nyu", help="(1) nyu (2) m3d")
    args = parser.parse_args()

    find_line(args.txt_list_path, args.dataset)
