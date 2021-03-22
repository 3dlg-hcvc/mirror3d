#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/scannet_130_anno_env_setup
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 1 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_error_100/with_mirror/precise \
--f 575 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_130_anno_env_setup.log" ::: {0..135}
