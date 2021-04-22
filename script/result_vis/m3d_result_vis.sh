#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_vis_0408
mkdir -p $log_folder
echo $log_folder

parallel -j 8 --eta "python visualization/result_visualization.py \
--stage 1 \
--multi_processing --process_index {1} \
--method_predFolder_txt /project/3dlg-hcvc/mirrors/www/notes/m3d_vis_0418.txt \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--f 537 \
--pred_w 640 \
--pred_h 512 \
--test_json /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--output_folder /project/3dlg-hcvc/mirrors/www/cr_vis/debug_m3d_0418 >& ${log_folder}/m3d_vis_0408.log" ::: {0..660}
