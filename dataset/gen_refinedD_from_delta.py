import argparse
import os
from utils.general_utlis import *
from tqdm import tqdm

def gen_refinedD_from_deltaD(unzipped_folder_path, mask_version):

    if "mp3d" in unzipped_folder_path:
        rawD_folder = os.path.join(unzipped_folder_path, "raw_meshD")
        delta_img_folder = os.path.join(unzipped_folder_path, "delta_depth_{}".format(mask_version))
        refD_folder = os.path.join(unzipped_folder_path, "refined_meshD_{}".format(mask_version))
        deltaD_list = [i.strip() for i in os.popen("find {} -type f".format(delta_img_folder)).readlines()]  
        os.makedirs(refD_folder, exist_ok=True)
        for one_delta_path in tqdm(deltaD_list):
            one_rawD_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "raw_meshD")
            refinedD = cv2.imread(one_rawD_path, cv2.IMREAD_ANYDEPTH) + cv2.imread(one_delta_path, cv2.IMREAD_ANYDEPTH)
            one_refD_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "refined_meshD_{}".format(mask_version))
            os.makedirs(os.path.split(one_refD_path)[0], exist_ok=True)
            cv2.imwrite(one_refD_path, refinedD.astype(np.uint16))

        print("refiend depth saved to:", refD_folder)

    rawD_folder = os.path.join(unzipped_folder_path, "raw_sensorD")
    delta_img_folder = os.path.join(unzipped_folder_path, "delta_depth_{}".format(mask_version))
    refD_folder = os.path.join(unzipped_folder_path, "refined_sensorD_{}".format(mask_version))
    deltaD_list = [i.strip() for i in os.popen("find {} -type f".format(delta_img_folder)).readlines()]
    os.makedirs(refD_folder, exist_ok=True)
    for one_delta_path in tqdm(deltaD_list):
        one_rawD_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "raw_sensorD")
        refinedD = cv2.imread(one_rawD_path, cv2.IMREAD_ANYDEPTH) + cv2.imread(one_delta_path, cv2.IMREAD_ANYDEPTH)
        one_refD_path = one_delta_path.replace("delta_depth_{}".format(mask_version), "refined_sensorD_{}".format(mask_version))
        os.makedirs(os.path.split(one_refD_path)[0], exist_ok=True)
        cv2.imwrite(one_refD_path, refinedD.astype(np.uint16))

    print("refiend depth saved to:", refD_folder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--mask_version', default="precise", help="2 mask version : precise/ coarse")
    parser.add_argument(
        '--unzipped_folder_path', default="")
    args = parser.parse_args()
    gen_refinedD_from_deltaD(args.unzipped_folder_path, args.mask_version)
