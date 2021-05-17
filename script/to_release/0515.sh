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



# saic test on refD checkpoint 0
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_refD_0.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final


# saic test on refD checkpoint 1
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_refD_1.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final


# saic test on refD checkpoint 2
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/saic_refD_2.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 537 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/m3d_final


echo "Enter the output file list.txt to refine: "  
# read to_ref_txt_list
to_ref_txt_list="waste/nyu_dg_1.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_1.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
    --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
    --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
    --coco_focal_len 519 \
    --depth_shift 1000 \
    --input_height 480 \
    --input_width 640 \
    --batch_size 8 \
    --checkpoint_save_freq 1500 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
    --ref_mode DE_border \
    --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
    --to_ref_txt $one_txt
done