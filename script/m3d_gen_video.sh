
#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/m3d_gen_video
mkdir -p $log_folder
echo $log_folder
parallel -j 20 --eta "python /local-scratch/jiaqit/exp/Mirror3D/visualization/dataset_visualization.py \
--stage 5 \
--view_mode topdown \
--data_main_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/precise --f 1074 \
--output_folder /local-scratch/jiaqit/exp/Mirror3D/dataset/m3d/with_mirror/precise \
--multi_processing --overwrite --process_index {1} >& ${log_folder}/m3d_gen_video.log" ::: {0..4150}