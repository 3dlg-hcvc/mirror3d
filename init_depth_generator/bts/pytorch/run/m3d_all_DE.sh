#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:v100l:2        # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node) 
#SBATCH --time=1-20:45           # time (DD-HH:MM) # TODO set a small number to test first
#SBATCH --job-name=m3d_refineD_mesh_D # TODO
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate bts
python bts_main.py --num_epochs 200  \
--model_name m3d_refineD_mesh_D \
--coco_input \
--learning_rate 1e-5 \
--refined_depth False \
--coco_focal_len 269 \
--depth_shift 4000 \
--learning_rate 1e-5 \
--eval_freq 8000 \
--refined_depth True \
--mesh_depth True \
--input_height 256 \
--input_width 320 \
--batch_size 1 \
--coco_root /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip \
--coco_root /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip \
--coco_train /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip/coco_input/DE_1280_1024_all/pos_train_only_DE.json \
--coco_val /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip/coco_input/DE_1280_1024_all/pos_val_only_DE.json \
|tee -a /project/3dlg-hcvc/jiaqit/waste/bts_m3d_refineD_mesh_D_$(date +%Y-%m-%d-%H:%M:%S).log

