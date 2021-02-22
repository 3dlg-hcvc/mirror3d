#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:4              # Number of GPUs (per node)
#SBATCH --mem=128000               # memory (per node)
#SBATCH --time=0-20:20            # time (DD-HH:MM)
#SBATCH --job-name=nyu_test
#SBATCH --output=/home/jiaqit/projects/rrg-msavva/jiaqit/result/log/%x-%j.out
source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate
conda activate mirror
python bts_main.py --batch_size 16  --model_name nyu | tee -a /home/jiaqit/projects/rrg-msavva/jiaqit/result/log/nyu_bts_$(date +%Y-%m-%d-%H:%M:%S).log
