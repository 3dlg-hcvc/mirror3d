#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:4         # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=1-20:45             # time (DD-HH:MM) # TODO set a small number to test first
#SBATCH --job-name=s3d_allP50_m3d500 # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror
python bts_main.py --num_epochs 200  \
--model_name s3d_allP50_m3d500 \
--coco_input \
--coco_focal_len 320,320,320,550,1076 \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_low_percentage,/home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/img_640_480_ransac_f550,/home/jiaqit/projects/rrg-msavva/jiaqit/data/detph_estimation/matterport_DE \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_final_0924,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/mirror_pers_low_percentage,/home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/img_640_480_ransac_f550,/home/jiaqit/projects/rrg-msavva/jiaqit/data/detph_estimation/matterport_DE \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/complete_pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/incomplete_pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/low_percentage/top50/incomplete_pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/input_info/img_640_480_ransac_f550_bm_split/pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/detph_estimation/m3d_top500/pos_train_only_DE.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/complete_pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/high_percentage/top50/incomplete_pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/s3d_mirror/coco_input/low_percentage/top50/incomplete_pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/input_info/img_640_480_ransac_f550_bm_split/pos_val_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/detph_estimation/m3d_top500/pos_val_only_DE.json  |\
tee -a /home/jiaqit/projects/rrg-msavva/jiaqit/result/log/s3d_allP50_m3d500_$(date +%Y-%m-%d-%H:%M:%S).log