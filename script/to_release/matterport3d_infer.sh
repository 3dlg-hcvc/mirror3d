# echo "PlaneRCNN rawD inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/planercnn_rawD.pth \
# --config mirror3dnet/config/planercnn_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_focal_len 574 \
# --depth_shift 4000 \
# --input_height 512 \
# --input_width 640 \
# --batch_size 1 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory cr_output/m3d \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# echo "PlaneRCNN refD inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/planercnn_refD.pth \
# --config mirror3dnet/config/planercnn_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_focal_len 574 \
# --depth_shift 4000 \
# --input_height 512 \
# --input_width 640 \
# --batch_size 1 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory cr_output/m3d \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# echo "Mirror3DNet rawD inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_rawD.pth \
# --config mirror3dnet/config/mirror3dnet_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_focal_len 574 \
# --depth_shift 4000 \
# --input_height 512 \
# --input_width 640 \
# --batch_size 1 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory cr_output/m3d \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# echo "Mirror3DNet refD inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_refD.pth \
# --config mirror3dnet/config/mirror3dnet_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
# --coco_focal_len 574 \
# --depth_shift 4000 \
# --input_height 512 \
# --input_width 640 \
# --batch_size 1 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory cr_output/m3d \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet 5 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_5.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_5_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_5_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory cr_output/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_5.npy


echo "Mirror3DNet 7 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_7.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_7_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_7_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory cr_output/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_7.npy

echo "Mirror3DNet 12 normal inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_12.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_12_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_12_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory cr_output/m3d \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_12.npy