
#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/set_up_anno900
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 1 \
--data_main_folder /local-scratch/jiaqit/exp/annotation/m3d_900_anno --f 1076 \
--multi_processing --process_index {1} >& ${log_folder}/set_up_anno900.log" ::: {0..1000}