import argparse
import os
import cv2
import numpy as np
from mirror3d.utils.general_utils import *
from tqdm import tqdm


def gen_refinedD_from_deltaD(unzipped_folder_path, mask_version):
    if "mp3d" in unzipped_folder_path:
        raw_d_folder = os.path.join(unzipped_folder_path, "raw_meshD")
        delta_img_folder = os.path.join(unzipped_folder_path, "delta_depth_{}".format(mask_version))
        ref_d_folder = os.path.join(unzipped_folder_path, "refined_meshD_{}".format(mask_version))
        delta_d_list = [i.strip() for i in os.popen("find {} -type f".format(delta_img_folder)).readlines()]
        os.makedirs(ref_d_folder, exist_ok=True)
        for one_delta_path in tqdm(delta_d_list):
            one_raw_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "raw_meshD")
            refined_d = cv2.imread(one_raw_d_path, cv2.IMREAD_ANYDEPTH) + cv2.imread(one_delta_path,
                                                                                     cv2.IMREAD_ANYDEPTH)
            one_ref_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version),
                                                    "refined_meshD_{}".format(mask_version))
            os.makedirs(os.path.split(one_ref_d_path)[0], exist_ok=True)
            cv2.imwrite(one_ref_d_path, refined_d.astype(np.uint16))
        print("refined depth saved to:", ref_d_folder)

        raw_d_folder = os.path.join(unzipped_folder_path, "raw_sensorD")
        delta_img_folder = os.path.join(unzipped_folder_path, "delta_depth_{}".format(mask_version))
        ref_d_folder = os.path.join(unzipped_folder_path, "refined_sensorD_{}".format(mask_version))
        delta_d_list = [i.strip() for i in os.popen("find {} -type f".format(delta_img_folder)).readlines()]
        os.makedirs(ref_d_folder, exist_ok=True)
        for one_delta_path in tqdm(delta_d_list):
            one_mask_path = rreplace(one_delta_path.replace("delta_depth_{}".format(mask_version), "mirror_instance_mask_{}".format(mask_version)), "d", "i")
            mirror_mask = cv2.imread(one_mask_path, cv2.IMREAD_ANYDEPTH)
            one_raw_sensor_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "raw_sensorD")
            one_ref_mesh_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version),
                                        "refined_meshD_{}".format(mask_version))
            refined_d = cv2.imread(one_raw_sensor_d_path, cv2.IMREAD_ANYDEPTH)*((mirror_mask==0).astype(np.uint8)) + \
                        cv2.imread(one_ref_mesh_d_path, cv2.IMREAD_ANYDEPTH)*((mirror_mask>0).astype(np.uint8))
            one_ref_sensor_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version),
                                                    "refined_sensorD_{}".format(mask_version))
            os.makedirs(os.path.split(one_ref_sensor_d_path)[0], exist_ok=True)
            cv2.imwrite(one_ref_sensor_d_path, refined_d.astype(np.uint16))
        print("refined depth saved to:", ref_d_folder)
    else:
        raw_d_folder = os.path.join(unzipped_folder_path, "raw_sensorD")
        delta_img_folder = os.path.join(unzipped_folder_path, "delta_depth_{}".format(mask_version))
        ref_d_folder = os.path.join(unzipped_folder_path, "refined_sensorD_{}".format(mask_version))
        delta_d_list = [i.strip() for i in os.popen("find {} -type f".format(delta_img_folder)).readlines()]
        os.makedirs(ref_d_folder, exist_ok=True)
        for one_delta_path in tqdm(delta_d_list):
            one_raw_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "raw_sensorD")
            refined_d = cv2.imread(one_raw_d_path, cv2.IMREAD_ANYDEPTH) + cv2.imread(one_delta_path, cv2.IMREAD_ANYDEPTH)
            one_ref_d_path = one_delta_path.replace("delta_depth_{}".format(mask_version),
                                                    "refined_sensorD_{}".format(mask_version))
            os.makedirs(os.path.split(one_ref_d_path)[0], exist_ok=True)
            cv2.imwrite(one_ref_d_path, refined_d.astype(np.uint16))
        print("refined depth saved to:", ref_d_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--mask_version', default="all", help="mask version : precise/ coarse/ all")
    parser.add_argument(
        '--unzipped_folder_path', default="")
    args = parser.parse_args()
    if args.mask_version == "all":
        gen_refinedD_from_deltaD(args.unzipped_folder_path, "precise")
        gen_refinedD_from_deltaD(args.unzipped_folder_path, "coarse")
    elif args.mask_version == "precise" or args.mask_version == "coarse":
        gen_refinedD_from_deltaD(args.unzipped_folder_path, args.mask_version)
    else:
        print("invalid mask version: (1) all : precise and coarse (2) precise (3) coarse")

