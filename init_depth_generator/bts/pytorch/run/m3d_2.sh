#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:4         # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=1-20:45            # time (DD-HH:MM) # TODO set a small number to test first
#SBATCH --job-name=m3d_top500 # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate bts_new
python tools/train_matterport.py \
--snapshot_period 50 \
--depth_shift 4000 \
--epoch 200 \
--input_width 160 \
--input_height 128 \
--train_batch_size 32 \
--train_coco_path /home/jiaqit/scratch/data/coco_input/debug/pos_train_only_DE.json \
--val_coco_path /home/jiaqit/scratch/data/coco_input/debug/pos_val_only_DE.json \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/m3d_unzip \
--refined_D False