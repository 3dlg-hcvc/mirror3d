#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_da
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
--stage 7 \
--data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
--output_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise \
--multi_processing --process_index {1} >& ${log_folder}/m3d_da.log" ::: {0..5000}

