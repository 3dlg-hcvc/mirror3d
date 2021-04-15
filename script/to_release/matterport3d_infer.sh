
# saic test on rawD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/finetune_dg/saic_rawD.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder cr_output/m3d_infer

# saic test on refD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/finetune_dg/saic_refD.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--output_save_folder cr_output/m3d_infer


# other depth + Mirror3dNet
echo "Enter the output file list.txt to refine: "  
# read to_ref_txt_list
to_ref_txt_list="waste/m3d_dg.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_10.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
    --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
    --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d \
    --coco_focal_len 574 \
    --depth_shift 4000 \
    --input_height 512 \
    --input_width 640 \
    --batch_size 8 \
    --checkpoint_save_freq 1500 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --log_directory cr_output/m3d_infer \
    --ref_mode DE_border \
    --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
    --to_ref_txt $one_txt
done

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
# --log_directory cr_output/m3d_infer \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet rawD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_rawD.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
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
--log_directory cr_output/m3d_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet refD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_refD.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_10_normal_mirror.json \
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
--log_directory cr_output/m3d_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# echo "Mirror3DNet 3 normal inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_3.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_3_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_3_normal_mirror.json \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_3.npy

# echo "Mirror3DNet 5 normal inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_5.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_5_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_5_normal_mirror.json \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_5.npy


# echo "Mirror3DNet 7 normal inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_7.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_7_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_7_normal_mirror.json \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_7.npy

# echo "Mirror3DNet 10 normal inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_10.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# echo "Mirror3DNet 12 normal inference"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --mesh_depth \
# --eval_save_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_12.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/train_12_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/m3d/with_mirror/precise/network_input_json/val_12_normal_mirror.json \
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
# --log_directory cr_output/m3d_infer \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_12.npy



# rawD + Mirror3dNet
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_10.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 574 \
--depth_shift 4000 \
--input_height 512 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory cr_output/m3d_infer \
--ref_mode rawD_border