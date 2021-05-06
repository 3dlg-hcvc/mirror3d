#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-10:45
#SBATCH --job-name=2_bts_refD_scratch_all
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory debug/nyu
