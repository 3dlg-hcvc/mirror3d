# bts train on refined sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d

# bts train on raw sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d

# vnl train on refined sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d


# vnl train on raw sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d

# saic train on refined sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--refined_depth \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 2 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d


# saic train on raw sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_all.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/mp3d


# planercnn on refined sensor depth
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_mirror.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d


# planercnn on raw sensor depth
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_mirror.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d



# mirror3dnet on refined sensor depth 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_mirror.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d


# mirror3dnet on raw sensor depth 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_mirror.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d

# mirror3dnet only normal 10 anchor normal
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train ../network_input_json/mp3d/train_10_precise_normal_mirror.json \
--coco_val ../network_input_json/mp3d/val_10_precise_normal_mirror.json \
--coco_train_root ../dataset/mp3d \
--coco_val_root ../dataset/mp3d \
--coco_focal_len 537 \
--mesh_depth \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--resume_checkpoint_path ./checkpoint/R-50.pkl \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/mp3d