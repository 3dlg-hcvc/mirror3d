#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-02:45
#SBATCH --job-name=val_bts_test_on_rawD_checkpoint_1
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/bts_rawD_1 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/m3d_final1_raw