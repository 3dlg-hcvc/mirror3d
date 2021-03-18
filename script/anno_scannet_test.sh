
#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/set_up_anno_scannet_test
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 1 \
--data_main_folder /local-scratch/share/annotation_data/scannet/test --f 575  --overwrite \
--multi_processing --process_index {1} >& ${log_folder}/set_up_anno_scannet_test.log" ::: {0..130}