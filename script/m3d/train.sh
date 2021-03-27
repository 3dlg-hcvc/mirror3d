# bts train on refined depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output

# bts train on sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output

# vnl train on refined depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output


# vnl train on sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output

# saic train on refined depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output


# saic train on sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--mesh_depth \
--log_directory output



# planercnn on refD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--log_directory output


# planercnn on rawD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--log_directory output


# mirror3dnet on refD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--log_directory output


# mirror3dnet on rawD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--log_directory output



# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--log_directory output

# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_3_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_3_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_3.npy \
--log_directory output

# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_5_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_5_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_5.npy \
--log_directory output

# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_7_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_7_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_7.npy \
--log_directory output

# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_12_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_12_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--anchor_normal_npy /project/6049211/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_12.npy \
--log_directory output

