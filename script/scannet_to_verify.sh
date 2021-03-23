#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/scannet_final
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 3 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--f 575 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_final.log" ::: {0..3000}


parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 4 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--specific_clamp_parameter_json /project/3dlg-hcvc/mirrors/www/final_anno/scannet_1900/imgNmae_expand_range.json \
--f 575 --expand_range 100 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_final.log" ::: {0..3000}


parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
--stage all \
--data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--f 575 --multi_processing --overwrite --process_index {1} >& ${log_folder}/scannet_final.log" ::: {0..3000}