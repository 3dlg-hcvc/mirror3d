#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:2         # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=0-0:45            # time (DD-HH:MM) # TODO set a small number to test first
#SBATCH --job-name=m3d_refined_D # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate bts
python bts_main.py --num_epochs 200  \
--model_name m3d_refined_D \
--coco_input \
--refined_depth True \
--coco_focal_len 538 \
--coco_root /project/rrg-msavva/jiaqit/data/m3d_data/data/m3d_all_640_480_ransac \
--coco_root /project/rrg-msavva/jiaqit/data/m3d_data/data/m3d_all_640_480_ransac \
--coco_train /project/rrg-msavva/jiaqit/data/m3d_data/coco_input/m3d_all_640_480_ransac/pos_train_normalFormat_3_normal.json \
--coco_val /project/rrg-msavva/jiaqit/data/m3d_data/coco_input/m3d_all_640_480_ransac/pos_val_normalFormat_3_normal.json \
|tee -a /home/jiaqit/projects/rrg-msavva/jiaqit/result/log/bts_m3d_refined_D_$(date +%Y-%m-%d-%H:%M:%S).log

