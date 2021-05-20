#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=48000
#SBATCH --time=0-01:45
#SBATCH --job-name=nyu_complete_nyu_rawD_plus_Mirror3dNet_random
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror3d
export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_normal_rawD_resume_2021-05-08-00-16-02/model_0014999.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
--ref_mode rawD_border
