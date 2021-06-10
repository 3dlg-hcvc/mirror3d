"""This is a tool to extract rgb/depth image from .mat file."""

import argparse
import os
import h5py
import numpy as np
from PIL import Image

def export_img_from_mat(mat_path, output_dir):
    """Export rgb/depth image from mat file.
    Args:
        mat_path (str): the path of .mat file.
                        Assume the depth info is in the "depths" field, the rgb info is in the "images" field.
        output_dir (str): the path to save the output image.

    """
    h5_file = h5py.File(mat_path)
    # center crop the NYUv2 image by 5%
    new_w=608
    new_h=456
    ori_w=640
    ori_h=480
    w_border = int((ori_w - new_w)/2)
    h_border = int((ori_h - new_h)/2)

    # exporting color image
    img_data = h5_file["images"]
    img_np = np.array(img_data)
    img_np = np.uint8(img_np).transpose((0, 3, 2, 1))
    color_output_dir = os.path.join(output_dir, "color")
    os.makedirs(color_output_dir, exist_ok=True)

    for i, single_img_np in enumerate(img_np):
        single_img_np = single_img_np[h_border:h_border+new_h, w_border:w_border+new_w]
        img_pil = Image.fromarray(single_img_np)
        save_path = os.path.join(color_output_dir, '{}.jpg'.format(str(i+1)))
        img_pil.save(save_path)
    print("color images saved under : ", color_output_dir)

    # exporting depth image (unit mm)
    img_data = h5_file["depths"]
    img_np = np.array(img_data)
    # convert the depth unit to mm
    img_np = img_np * 1000
    img_np = np.uint16(img_np).transpose((0, 2, 1))
    depth_output_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_output_dir, exist_ok=True)

    for i, single_img_np in enumerate(img_np):
        single_img_np = single_img_np[h_border:h_border+new_h, w_border:w_border+new_w]
        img_pil = Image.fromarray(single_img_np)
        save_path = os.path.join(depth_output_dir, '{}.png'.format(str(i+1)))
        img_pil.save(save_path)

    print("depth images saved under : ", depth_output_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True, help="the mat file path")
    parser.add_argument("--output_dir", default="nyu", help="the output directory; default saved under ./nyu")

    args = parser.parse_args()
    export_img_from_mat(args.mat_path, args.output_dir)
