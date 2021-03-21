
#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_clamp
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 4 \
--data_main_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/precise --f 1074 \
--anno_output_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/precise \
--multi_processing --clamp_dis 400 --process_index {1} >& ${log_folder}/m3d_clamp.log" ::: {0..4150}