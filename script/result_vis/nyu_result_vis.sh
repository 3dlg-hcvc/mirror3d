#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/nyu_vis_0518
mkdir -p $log_folder
echo $log_folder
parallel -j 8 --eta "python visualization/result_visualization.py \
--stage 1 \
--multi_processing  \
--method_predFolder_txt /project/3dlg-hcvc/mirrors/www/notes/nyu_vis_0518.txt \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--f 519 \
--overwrite \
--pred_w 640 \
--pred_h 480 \
--output_folder /project/3dlg-hcvc/mirrors/www/final_result_vis/nyu_result_vis --process_index {1} >& ${log_folder}/nyu_vis_0518.log" ::: {0..70}
