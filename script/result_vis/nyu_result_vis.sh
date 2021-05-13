#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/nyu_vis_0512
mkdir -p $log_folder
echo $log_folder
parallel -j 8 --eta "python waste/result_vis_temp.py \
--stage 1 \
--multi_processing  \
--method_predFolder_txt /project/3dlg-hcvc/mirrors/www/notes/nyu_vis_0418.txt \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--f 519 \
--overwrite \
--pred_w 640 \
--pred_h 480 \
--output_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_result_vis --process_index {1} >& ${log_folder}/nyu_vis_0512.log" ::: {0..70}
