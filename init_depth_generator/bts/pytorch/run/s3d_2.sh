#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:4         # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=1-20:45            # time (DD-HH:MM) # TODO set a small number to test first
#SBATCH --job-name=s3d_allP_top50 # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror
python bts_main.py --num_epochs 200  \
--model_name s3d_allP_top50 \
--coco_input \
--coco_focal_len 320,320,320 \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_low_percentage \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_low_percentage \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/complete_pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/incomplete_pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/low_percentage/top50/incomplete_pos_train_normalFormat_3_normal.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/complete_pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/incomplete_pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/low_percentage/top50/incomplete_pos_val_normalFormat_3_normal.json  |\
tee -a /home/jiaqit/projects/rrg-msavva/jiaqit/result/log/s3d_allP_top50_$(date +%Y-%m-%d-%H:%M:%S).log