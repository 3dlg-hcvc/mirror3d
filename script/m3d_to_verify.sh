#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_0_50_re_ll_result
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/waste/result_vis_temp.py \
--stage 3 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/m3d_0_50_ll/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/final_anno/m3d_0_50_ll/with_mirror/precise \
--f 1076 --multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_0_50_re_ll_result.log" ::: {0..300}


parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
--stage all \
--data_main_folder /project/3dlg-hcvc/mirrors/www/final_anno/m3d_0_50_ll/with_mirror/precise \
--output_folder /project/3dlg-hcvc/mirrors/www/final_anno/m3d_0_50_ll/with_mirror/precise \
--f 1076 --multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_0_50_re_ll_result.log" ::: {0..300}