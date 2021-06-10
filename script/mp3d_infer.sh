# bts infer on refined mesh depth
python mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/mp3d/bts_refD \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--output_save_folder ../output/mp3d

# bts infer on raw mesh depth
python mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/mp3d/bts_rawD \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--output_save_folder ../output/mp3d


# vnl infer on refined mesh depth
python mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/mp3d/vnl_refD.pth \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--batch_size 4 \
--output_save_folder ../output/mp3d

# vnl infer on raw mesh depth
python mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/mp3d/vnl_rawD.pth \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--batch_size 4 \
--output_save_folder ../output/mp3d


# saic infer on refined mesh depth
python mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/mp3d/saic_refD.pth \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--batch_size 2 \
--output_save_folder ../output/mp3d


# saic infer on raw mesh depth
python mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/saic_rawD.pth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--output_save_folder ../output/mp3d


# planercnn on refined mesh depth
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path ../checkpoint/mp3d/planercnn_refD.pth \
--config mirror3d/mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d


# planercnn on raw mesh depth
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/planercnn_rawD.pth \
--config mirror3d/mirror3dnet/config/planercnn_config.yml \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d



# mirror3dnet on refined mesh depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_refD.pth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d


# mirror3dnet on raw mesh depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_rawD.pth \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d

# mirror3dnet only normal 10 anchor normal
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_normal_10.pth \
--config mirror3d/mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_val ../mirror3d_input/mp3d/val_10_precise_normal_mirror.json \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 1074 \
--mesh_depth \
--depth_shift 4000 \
--input_height 1024 \
--input_width 1280 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d