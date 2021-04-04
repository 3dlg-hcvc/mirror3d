#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-23:45
#SBATCH --job-name=vnl_rawD_resume_mirror
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/m3d \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/vnl/vnl_m3d_meshD_rawD_2021-04-02-17-42-25/checkpoint/epoch0_step30000.pth \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory debug
