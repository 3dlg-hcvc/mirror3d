import os
import time
import datetime
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.Mirror3D_eval import Mirror3d_eval

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)


def save_txt(save_path, data):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", save_path, len(data))



def inference(
        args, model, test_loaders, metrics, save_dir="", logger=None
):
    mirror3d_eval = Mirror3d_eval(args.refined_depth, logger, Input_tag="RGBD", method_tag="saic")

    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch = model.preprocess(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = model.postprocess(pred)

                try:
                    metrics_meter.update(post_pred, batch["gt_depth"])
                except:
                    print("error !!!!!!!!!!! ", batch["gt_depth_path"][0])
                    continue
                gt_depth_path = batch["gt_depth_path"][0]
                if gt_depth_path.find("m3d") > 0:
                    mask_path = rreplace(gt_depth_path.replace("hole_refined_depth", "instance_mask"), "d", "i")
                else:
                    mask_path = gt_depth_path.replace("hole_refined_depth", "instance_mask")
                if not os.path.exists(mask_path) or gt_depth_path.find("hole_refined_depth") < 0:
                    mask_path = None
                    raw_path = None
                else:
                    raw_path = mask_path.replace("instance_mask", "raw")
                # print(mask_path)
                

                post_pred = model.predict_process(post_pred, batch, args.depth_shift)
                post_pred = post_pred.cpu().numpy()
                post_pred[post_pred<0] = 0

                gt_depth = cv2.resize(cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH), (post_pred.shape[1], post_pred.shape[0]), 0, 0, cv2.INTER_NEAREST) / args.depth_shift
                mirror3d_eval.compute_and_update_mirror3D_metrics(post_pred/args.depth_shift, args.depth_shift, batch["color_img_path"][0])
                mirror3d_eval.save_result(args.log_directory, post_pred/args.depth_shift, args.depth_shift, batch["color_img_path"][0])
                
        mirror3d_eval.print_mirror3D_score()
        

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)
