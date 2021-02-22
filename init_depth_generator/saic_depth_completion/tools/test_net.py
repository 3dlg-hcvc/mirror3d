import torch

import argparse

from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.data.datasets.nyuv2_test import NyuV2Test
from saic_depth_completion.engine.inference import inference
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel

def main():
    parser = argparse.ArgumentParser(description="Some training params.")

    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="LRN", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="/local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/configs/lrn/LRN_efficientnet-b4_lena.yaml", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--save_dir", default="/project/3dlg-hcvc/jiaqit/waste", type=str, help="Save dir for predictions"
    )
    parser.add_argument( # TODO
        "--depth_shift", default=1000, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument(
        "--weights", default="/project/3dlg-hcvc/jiaqit/checkpoint/cvpr2021/saic_other/refD/snapshot_49_110_all.pth", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--coco_path", default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json", type=str, help="coco format json input path" # DE_1280_1024_all_1000debug debug_test
    )
    parser.add_argument(
        "--coco_root", default="/local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608", type=str, help="coco format data root"
    )
    parser.add_argument(
        "--refined_depth", default=True, type=bool, help="use refined normal map/ noisy normal map"
    )
    parser.add_argument( # TODO
        "--input_height", default=480, type=int, help="depth shift to meter from GT depth"
    )
    parser.add_argument( # TODO
        "--input_width", default=640, type=int, help="depth shift to meter from GT depth"
    )

    args = parser.parse_args()

    print("################# check point ##################")
    print(args.weights)

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()

    snapshoter = Snapshoter(model, logger=logger)
    snapshoter.load(args.weights)

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

    test_datasets = {
        "test_matterport": Matterport(root=args.coco_root, coco_path=args.coco_path, refined_depth=args.refined_depth,split="test",args=args),

    }
    test_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate
        )
        for k, v in test_datasets.items()
    }

    inference(
        model,
        test_loaders,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()