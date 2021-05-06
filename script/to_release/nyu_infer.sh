# bts test on official checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/nyu/bts_official_nyu_raw \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer


# bts test on refD checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/nyu/val_converge/bts_refD \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer


# vnl test on official checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/nyu/vnl_official_nyu_raw.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer


vnl test on refD checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/nyu/val_converge/vnl_refD.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer


# saic test on rawD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/nyu/val_converge/saic_rawD.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer

# saic test on refD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/nyu/val_converge/saic_refD.pth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer

# # other depth + Mirror3dNet
# echo "Enter the output file list.txt to refine: "  
# # read to_ref_txt_list
# to_ref_txt_list="waste/nyu_dg.txt"
# for one_txt in $(cat $to_ref_txt_list); do
#     echo "applied Mirror3dNet on $one_txt"
#     python mirror3dnet/run_mirror3dnet.py \
#     --eval \
#     --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_10.pth \
#     --config mirror3dnet/config/mirror3dnet_normal_config.yml \
#     --refined_depth \
#     --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
#     --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
#     --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
#     --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
#     --coco_focal_len 519 \
#     --depth_shift 1000 \
#     --input_height 480 \
#     --input_width 640 \
#     --batch_size 8 \
#     --checkpoint_save_freq 1500 \
#     --num_epochs 100 \
#     --learning_rate 1e-4 \
#     --log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
#     --ref_mode DE_border \
#     --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
#     --to_ref_txt $one_txt
# done

echo "rawD + Mirror3dNet"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_noraml_10.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
--ref_mode rawD_border




echo "PlaneRCNN rawD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/planercnn_rawD.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "PlaneRCNN refD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/planercnn_refD.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet rawD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_rawD.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet refD inference"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/m3d/m3d_refD.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/debug_1st/nyu_infer \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy
