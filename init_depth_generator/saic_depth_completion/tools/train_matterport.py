import torch
import os
import argparse

import numpy as np
from torchvision import transforms
from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.engine.train import train
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.utils.tracker import ComposedTracker, Tracker
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel


def main():
    parser = argparse.ArgumentParser(description="Some training params.")
    parser.add_argument(
        "--debug", dest="debug", type=bool, default=False, help="Setup debug mode"
    )
    parser.add_argument(
        "--postfix", dest="postfix", type=str, default="", help="Postfix for experiment's name"
    )
    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="LRN", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="./configs/lrn/LRN_efficientnet-b4_lena.yaml", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument( # TODO
        "--snapshot_period", default=1000, type=int, help="Snapshot model one time over snapshot period"
    )
    parser.add_argument( # TODO
        "--depth_shift", default=4000, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument( # TODO
        "--epoch", default=50, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument( # TODO
        "--train_batch_size", default=2, type=int, 
    )
    parser.add_argument( # TODO
        "--test_batch_size", default=2, type=int, 
    )
    parser.add_argument( # TODO
        "--input_height", default=512, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument( # TODO
        "--input_width", default=640, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument( # TODO
        "--train_coco_path", default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_train_normalFormat_10_normal.json", type=str, help="coco format json input path"
    )
    parser.add_argument( # TODO
        "--val_coco_path", default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json", type=str, help="coco format json input path"
    )
    parser.add_argument( # TODO
        "--coco_root", default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608", type=str, help="coco format data root"
    )
    parser.add_argument( # TODO
        "--refined_depth", default=False, type=bool, help="use refined normal map/ noisy normal map"
    )
    parser.add_argument(
        "--weights", default="/local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/model/lrn_b4.pth", type=str, metavar="FILE", help="path to config file"
    )

    args = parser.parse_args()
    

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()
    experiment = setup_experiment(
        cfg, args.config_file, logger=logger, training=True, debug=args.debug, postfix=args.postfix
    )

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((args.input_width,args.input_height)),transforms.ToTensor()])

    config_save_path = os.path.join(experiment.snapshot_dir, "setting.txt")

    with open(config_save_path, "w") as file:
        for item in args.__dict__.items():
            file.write("--{} {}".format(item[0],item[1]))
            file.write("\n")
    print("setting saved to : ", config_save_path)

    optimizer  = torch.optim.Adam(
        params=model.parameters(), lr=cfg.train.lr
    )
    if not args.debug:
        print("result saved to :",experiment.snapshot_dir)
        snapshoter = Snapshoter(
            model, optimizer, period=args.snapshot_period, logger=logger, save_dir=experiment.snapshot_dir
        )
        tensorboard = Tensorboard(experiment.tensorboard_dir)
        tracker = ComposedTracker([
            Tracker(subset="test_matterport", target="mse", snapshoter=snapshoter, eps=0.01),
            Tracker(subset="val_matterport", target="mse", snapshoter=snapshoter, eps=0.01),
        ])
    else:
        snapshoter, tensorboard, tracker = None, None, None

    if os.path.exists(args.weights):
        snapshoter.load(args.weights)
        print("############### loading checkpoint : ", args.weights)
        logger.info("############### loading checkpoint : ", args.weights)
    else:
        print("############### error loading : ", args.weights)
        logger.info("############### error loading : ", args.weights)

    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
        'ssim': SSIM(),
    }

    train_dataset = Matterport(root=args.coco_root, coco_path=args.train_coco_path, refined_depth=args.refined_depth, args=args, transforms=transform, split="train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate
    )

    val_datasets = {
        "val_matterport": Matterport(root=args.coco_root, coco_path=args.val_coco_path, refined_depth=args.refined_depth, args=args, transforms=transform, split="val"),
        # "test_matterport": Matterport(root=args.coco_root, coco_path=args.coco_path, refined_depth=args.refined_depth,split="test"),
    }
    val_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=default_collate
        )
        for k, v in val_datasets.items()
    }

    train(
        model,
        train_loader,
        val_loaders=val_loaders,
        optimizer=optimizer,
        snapshoter=snapshoter,
        epochs=args.epoch,
        logger=logger,
        metrics=metrics,
        tensorboard=tensorboard,
        tracker=tracker
    )


if __name__ == "__main__":
    main()