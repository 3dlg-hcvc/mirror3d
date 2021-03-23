#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_final_3800
mkdir -p $log_folder
echo $log_folder

# parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
# --stage 8 \
# --color_img_list /project/3dlg-hcvc/mirrors/www/Mirror3D_final/other_info/m3d_all_color_img_list.txt \
# --data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
# --anno_output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
# --f 1076 --multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_final_3800.log" ::: {0..4200}

parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 3 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
--anno_output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
--f 1076 --multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_final_3800.log" ::: {1999..4000}


# parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
# --stage all \
# --data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
# --output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
# --f 1076 --multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_final_3800.log" ::: {0..2000}