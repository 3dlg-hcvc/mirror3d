#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/scannet_vis_0408
mkdir -p $log_folder
echo $log_folder

parallel -j 8 --eta "python visualization/result_visualization.py \
--stage 1 \
--multi_processing --process_index {1} \
--method_predFolder_txt /project/3dlg-hcvc/mirrors/www/notes/scannet_vis_0503.txt \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--f 574 \
--pred_w 640 \
--pred_h 480 \
--test_json /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--output_folder /project/3dlg-hcvc/mirrors/www/cr_vis/scannet_result_vis >& ${log_folder}/scannet_vis_0408.log" ::: {0..350}
