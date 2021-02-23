import time
import datetime
import torch
from tqdm import tqdm

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from utils.Mirror3D_eval import Mirror3d_eval
import cv2

def validate(
        args, model, val_loaders, metrics, epoch=0, logger=None, tensorboard=None, tracker=None, final_result=False
):

    mirror3d_eval = Mirror3d_eval(args.refined_depth, logger, Input_tag="RGBD", method_tag="saic")

    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in val_loaders.items():
        logger.info(
            "Validate: ep: {}, subset -- {}. Total number of batches: {}.".format(epoch, subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch = model.preprocess(batch)
            pred = model(batch)

            

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                metrics_meter.update(post_pred, batch["gt_depth"])
                pred_depth = post_pred.squeeze().cpu()
                gt_depth_path = batch["gt_depth_path"][0]
                gt_depth = cv2.resize(cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST) / args.depth_shift
                mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth, args.depth_shift, batch["color_img_path"][0])
                if final_result:
                    mirror3d_eval.save_result(args.log_directory, pred_depth, args.depth_shift, batch["color_img_path"][0])

        mirror3d_eval.print_mirror3D_score()
        state = "Validate: ep: {}, subset -- {} | ".format(epoch, subset)
        logger.info(state + metrics_meter.suffix)

        metric_state = {k: v.global_avg for k, v in metrics_meter.meters.items()}

        if tensorboard is not None:
            tensorboard.update(metric_state, tag=subset, epoch=epoch)
            # tensorboard.add_figures(batch, post_pred, tag=subset, epoch=epoch)

        if tracker is not None:
            tracker.update(subset, metric_state)