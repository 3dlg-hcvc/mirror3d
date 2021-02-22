#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:p100:4         # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=0-00:10            # time (DD-HH:MM) # TODO
#SBATCH --job-name=m3d_top50 # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror
python bts_main.py --num_epochs 200 \
--model_name m3d_top50 \
--coco_input \
--coco_focal_len 550 \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/img_640_480_ransac_f550 \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/input_info/img_640_480_ransac_f550_bm_split/pos_train_normalFormat_3_normal.json,/home/jiaqit/projects/rrg-msavva/jiaqit/data/detph_estimation/m3d_top50/pos_train_only_DE.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/real_scene_mirror/matterport_0913/input_info/img_640_480_ransac_f550_bm_split/pos_val_normalFormat_3_normal.json  |\
tee -a /home/jiaqit/projects/rrg-msavva/jiaqit/result/log/m3d_top50_$(date +%Y-%m-%d-%H:%M:%S).log