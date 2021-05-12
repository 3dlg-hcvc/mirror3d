echo "saic test on rawD checkpoint 0"
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_rawD_0.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final


echo "saic test on rawD checkpoint 1"
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_rawD_1.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final


echo "saic test on rawD checkpoint 2"
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_rawD_2.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final