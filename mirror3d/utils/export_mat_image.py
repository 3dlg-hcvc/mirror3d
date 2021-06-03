"""This is a tool to extract rgb/depth image from .mat file."""

import argparse
import os
import h5py
import numpy as np
from PIL import Image


def export_img_from_mat(mat_path, output_dir, img_type="depths"):
    """Export rgb/depth image from mat file.
    Args:
        mat_path (str): the path of .mat file.
                        Assume the depth info is in the "depths" field, the rgb info is in the "images" field.
        output_dir (str): the path to save the output image.
        img_type (str): the image type rgb/depth to export.

    """
    h5_file = h5py.File(mat_path)
    img_data = h5_file[img_type]
    img_np = np.array(img_data)
    if img_type == "images":
        img_np = np.uint8(img_np).transpose((0, 3, 2, 1))
    elif img_type == "depths":
        img_np = img_np * 1000
        img_np = np.uint16(img_np).transpose((0, 2, 1))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for i, single_img_np in enumerate(img_np):
        img_pil = Image.fromarray(single_img_np)
        save_path = os.path.join(output_dir, str(i) + '.png')
        img_pil.save(save_path, 'PNG')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_path", required=True, help="the mat file path")
    ap.add_argument("--output_dir", required=True, help="the output directory")
    ap.add_argument("--img_type", required=True, help="the output image type images | depths")

    args = vars(ap.parse_args())
    export_img_from_mat(args["mat_path"], args["output_dir"], args["img_type"])
