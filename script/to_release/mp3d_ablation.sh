



echo "3 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_normal_3.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_3_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_3_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_3.npy \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--ref_mode none


echo "5 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_normal_5.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_5_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_5_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_5.npy \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--ref_mode none



echo "7 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_normal_7.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_7_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_7_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_7.npy \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--ref_mode none


echo "10 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_0.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--ref_mode none



echo "12 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_normal_12.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_12_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_12_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_12.npy \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--ref_mode none


echo "PlaneRCNN rawD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/planercnn_rawD_0.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--ref_mode none

# echo "PlaneRCNN refD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/planercnn_refD_0.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--ref_mode none

echo "Mirror3DNet rawD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_rawD_0.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--ref_mode none

echo "Mirror3DNet refD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_dt2_new/m3n_refD_0.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/m3d_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--ref_mode none