#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-06:45
#SBATCH --job-name=complete_saic_test_on_rawD_checkpoint_0
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final
