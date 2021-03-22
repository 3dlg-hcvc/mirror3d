#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/scannet_not_anno_re1_result
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 3 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--f 575 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_not_anno_re1_result.log" ::: {0..40}


parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 4 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--f 575 --expand_range 50 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_not_anno_re1_result.log" ::: {0..40}


parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
--stage all \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--output_folder /project/3dlg-hcvc/mirrors/www/final_anno/scannet_reclamp/with_mirror/precise \
--f 575 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_not_anno_re1_result.log" ::: {0..40}