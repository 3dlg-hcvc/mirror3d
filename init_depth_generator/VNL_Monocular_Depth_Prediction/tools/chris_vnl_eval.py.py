import os
import cv2
import torch
import numpy as np
from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.metric_depth_model import MetricDepthModel
from lib.core.config import cfg, merge_cfg_from_file
from lib.models.image_transfer import bins_to_depth
import json
import time
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
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

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

    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_folder = os.path.join(test_args.log_directory , time_tag)
    os.makedirs(output_folder, exist_ok=True)
    info_txt_save_path = os.path.join(output_folder, "vnl_mask_gtDepth_pDepth_npDepth.txt")
    if os.path.exists(info_txt_save_path):
        os.system('rm ' + info_txt_save_path)

    for info in tqdm(coco_val_info["images"]):
        img_path = os.path.join(test_args.dataroot, info["img_path"])
        with torch.no_grad():
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            _, pred_depth_softmax= model.module.depth_model(img_torch)
            pred_depth = bins_to_depth(pred_depth_softmax)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_scale = (pred_depth / pred_depth.max() *10000).astype(np.uint16)  # scale 60000 for visualization
            mask_path = img_path.replace("raw", "instance_mask")
            if not os.path.exists(mask_path):
                mask_path = None
            gt_depth_path = os.path.join(test_args.dataroot, info["mesh_refined_path"])
            depth_np_save_path = os.path.join(output_folder, gt_depth_path.split("/")[-1])
            with open(info_txt_save_path, "a") as file:
                file.write("{} {} {} {}".format(mask_path, gt_depth_path, depth_np_save_path, img_path))
                file.write("\n")
            pred_depth_scale[pred_depth_scale<0] = 0
            pred_depth_scale = pred_depth_scale.astype(np.uint16)
            cv2.imwrite(depth_np_save_path, pred_depth_scale)
    print("result save to : ", info_txt_save_path)
    print("checkpoint : ", test_args.load_ckpt)