#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-10:45
#SBATCH --job-name=1_m3n_full_rawD_resume_mirror
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d/m3d_rawD.pth \
--log_directory debug/nyu
