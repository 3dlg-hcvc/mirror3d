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
import time

def main():
    parser = argparse.ArgumentParser(description="Some training params.")


    # TODO
    parser.add_argument('--model_name',                type=str,   help='model name', default='vnl')
    #TODO
    parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
    #TODO
    parser.add_argument('--mesh_depth',                action='store_true',  help='using coco input format or not')
    # TODO
    parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json')
    # TODO
    parser.add_argument('--coco_train',                type=str,   help='coco json path', default='/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_train_normalFormat_10_normal.json')
    # TODO
    parser.add_argument('--coco_train_root',           type=str,   help='coco data root', 
        default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608")
    # TODO
    parser.add_argument('--coco_val_root',             type=str,   help='coco data root', 
        default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608")
    # TODO
    parser.add_argument('--coco_focal_len',            type=int,   help='nyu : 519', default=519)
    # TODO 
    parser.add_argument('--depth_shift',               type=int,   help='nyu : 1000, m3d : 4000', default=1000) # 4000 for m3d
    # TODO if coda boom
    parser.add_argument('--input_height',              type=int,   help='input height', default=256) # 480
    # TODO  if coda boom
    parser.add_argument('--input_width',               type=int,   help='input width',  default=320) # 640
    # TODO
    parser.add_argument('--batch_size',                type=int,   help='batch size',   dest='train_batch_size',  default=2)
    # TODO
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    # TODO
    parser.add_argument('--resume_checkpoint_path',    type=str,   help='path to a checkpoint to load', dest='weights', default="")
    # TODO
    parser.add_argument('--checkpoint_save_freq',      type=int,   help='Checkpoint saving frequency in global steps /iteration; nyu 5000; m3d 10000', dest='snapshot_period' , default=100)
    # TODO
    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='training output folder', default='/project/3dlg-hcvc/jiaqit/output')
    # TODO
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', dest='epoch', default=100)
    parser.add_argument('--output_save_folder',        type=str,   help='output_main_folder only use during inference', default='/project/3dlg-hcvc/jiaqit/exp_result')


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
        "--test_batch_size", default=1, type=int, 
    )


    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.train.lr = args.learning_rate
    cfg.freeze()

    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    dataset_name = ""
    if args.coco_train.find("nyu") > 0:
        dataset_name = "nyu"
    elif args.coco_train.find("m3d") > 0:
        dataset_name = "m3d"
    elif args.coco_train.find("scannet") > 0:
        dataset_name = "scannet"

    tag = ""
    if args.mesh_depth:
        tag = "meshD_"
    else:
        tag = "holeD_"

    if args.refined_depth:
        tag += "refinedD"
    else:
        tag += "rawD"

    args.model_name = "saic_{}_{}".format(dataset_name, tag)
    args.log_directory = os.path.join(args.log_directory, "saic","{}_{}".format(args.model_name, time_tag))
    os.makedirs(args.log_directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger(args.log_directory)
    experiment = setup_experiment(
        cfg, args.config_file, logger=logger, training=True, debug=args.debug, postfix=args.postfix, log_dir=args.log_directory
    )

    print("result saved to : {}".format(args.log_directory))
    logger.info("result saved to : {}".format(args.log_directory))

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((args.input_width,args.input_height)),transforms.ToTensor()])

    config_save_path = os.path.join(args.log_directory, "setting.txt")

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
        logger.info("############### loading checkpoint : {}".format(args.weights))


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

    train_dataset = Matterport(root=args.coco_train_root, coco_path=args.coco_train, refined_depth=args.refined_depth, args=args, transforms=transform, split="train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=default_collate
    )

    val_datasets = {
        "val_matterport": Matterport(root=args.coco_val_root, coco_path=args.coco_val, refined_depth=args.refined_depth, args=args, transforms=transform, split="val"),
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
        args,
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