# bts train on refined depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output

# bts train on sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output

# vnl train on refined depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output


# vnl train on sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output

# saic train on refined depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output


# saic train on sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output
