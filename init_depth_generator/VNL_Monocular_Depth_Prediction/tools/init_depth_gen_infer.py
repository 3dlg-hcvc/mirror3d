import os
import cv2
import torch
import numpy as np
from tools.parse_arg_test import TestOptions
from lib.core.config import cfg, merge_cfg_from_file
import json
import time
import logging

test_args = TestOptions().parse()
test_args.thread = 1
test_args.batchsize = 1
merge_cfg_from_file(test_args)
time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
output_folder = os.path.join(test_args.output_save_folder , "VNL_infer_{}".format(time_tag))
os.makedirs(output_folder, exist_ok=True)
cfg.TRAIN.LOG_DIR = output_folder

from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms

from data.load_dataset import CustomerDataLoader
from lib.models.metric_depth_model import MetricDepthModel
from lib.models.image_transfer import bins_to_depth
from utils.Mirror3D_eval import Mirror3d_eval



logger = setup_logging(__name__)
from tqdm import tqdm



def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info


if __name__ == '__main__':
    

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = MetricDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    coco_val_info = read_json(test_args.coco_val)
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    log_file_save_path = os.path.join(output_folder, "infer.log")
    logging.basicConfig(filename=log_file_save_path, filemode="a", level=logging.INFO, format=FORMAT)
    logging.info("output folder {}".format(output_folder))
    logging.info("checkpoint {}".format(test_args.load_ckpt))

    mirror3d_eval = Mirror3d_eval(test_args.refined_depth,logger=logging,Input_tag="RGB", method_tag="VNL")

    for info in tqdm(coco_val_info["images"]):
        img_path = os.path.join(test_args.coco_val_root, info["img_path"])
        with torch.no_grad():
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            _, pred_depth_softmax= model.module.depth_model(img_torch)
            pred_depth = bins_to_depth(pred_depth_softmax)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_scale = (pred_depth / pred_depth.max() *10000).astype(np.uint16)  # scale 60000 for visualization
            pred_depth_scale[pred_depth_scale<0] = 0
            pred_depth_scale = pred_depth_scale.astype(np.uint16)

            if test_args.refined_depth:
                if test_args.mesh_depth: # mesh refine
                    gt_depth_path = os.path.join(test_args.coco_val_root.strip().split(",")[0], info["mesh_refined_path"])
                else:  # hole refine
                    gt_depth_path = os.path.join(test_args.coco_val_root.strip().split(",")[0], info["hole_refined_path"])
            else:
                if test_args.mesh_depth: # mesh raw
                    gt_depth_path = os.path.join(test_args.coco_val_root.strip().split(",")[0], info["mesh_raw_path"])
                else:# mesh raw hole raw
                    gt_depth_path = os.path.join(test_args.coco_val_root.strip().split(",")[0], info["hole_raw_path"])   

            gt_depth = cv2.resize(cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth_scale.shape[1], pred_depth_scale.shape[0]), 0, 0, cv2.INTER_NEAREST)
            color_img_path = img_path
            mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth_scale / test_args.depth_shift, test_args.depth_shift, color_img_path)
            mirror3d_eval.save_result(output_folder, pred_depth_scale / test_args.depth_shift, test_args.depth_shift, color_img_path)

    mirror3d_eval.print_mirror3D_score()
    print("checkpoint : ", test_args.load_ckpt)