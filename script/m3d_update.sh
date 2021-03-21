
#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/course_ref
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py \
--stage 3 \
--data_main_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/coarse --f 1074 \
--anno_output_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/coarse \
--multi_processing --process_index {1} >& ${log_folder}/course_ref.log" ::: {0..4150}