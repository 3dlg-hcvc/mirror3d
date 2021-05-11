# # bts test on official checkpoint
# python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/nyu/bts_official_nyu_raw \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final


# # bts test on refD checkpoint 0
# python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/bts_refD_0 \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # bts test on refD checkpoint 1
# python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/bts_refD_1 \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # bts test on refD checkpoint 2
# python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/bts_refD_2 \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final


# # vnl test on official checkpoint
# python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
# --resume_checkpoint_path /project/3dlg-hcvc/jiaqit/debug_0331/checkpoint/nyu/vnl_official_nyu_raw.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final


# # vnl test on refD checkpoint 0
# python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/vnl_refD_0.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # vnl test on refD checkpoint 1
# python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/vnl_refD_1.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # vnl test on refD checkpoint 2
# python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/vnl_refD_2.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final


# # saic test on rawD checkpoint 0
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/val_converge/saic_rawD_0.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # saic test on rawD checkpoint 1
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/val_converge/saic_rawD_1.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # saic test on rawD checkpoint 2
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/val_converge/saic_rawD_2.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # saic test on refD checkpoint 0
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/saic_refD_0.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final


# # saic test on refD checkpoint 1
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/saic_refD_1.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# # saic test on refD checkpoint 2
# python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
# --refined_depth \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/nyu_final/saic_refD_2.pth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --output_save_folder /project/3dlg-hcvc/mirrors/www/final_result/nyu_final

# other depth + Mirror3dNet
echo "Enter the output file list.txt to refine: "  
# read to_ref_txt_list
to_ref_txt_list="waste/nyu_dg_0.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_0.pth \
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

echo "Enter the output file list.txt to refine: "  
# read to_ref_txt_list
to_ref_txt_list="waste/nyu_dg_2.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_2.pth \
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

# echo "rawD + Mirror3dNet 0"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_0.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --batch_size 8 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
# --ref_mode rawD_border


# echo "rawD + Mirror3dNet 1"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_1.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --batch_size 8 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
# --ref_mode rawD_border

# echo "rawD + Mirror3dNet 2"
# python mirror3dnet/run_mirror3dnet.py \
# --eval \
# --resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_normal_10_2.pth \
# --config mirror3dnet/config/mirror3dnet_normal_config.yml \
# --refined_depth \
# --coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
# --coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
# --coco_train_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --coco_val_root /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
# --anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
# --coco_focal_len 519 \
# --depth_shift 1000 \
# --input_height 480 \
# --input_width 640 \
# --batch_size 8 \
# --checkpoint_save_freq 1500 \
# --num_epochs 100 \
# --learning_rate 1e-4 \
# --log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
# --ref_mode rawD_border

echo "PlaneRCNN rawD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_rawD_0.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


echo "PlaneRCNN rawD inference 1"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_rawD_1.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "PlaneRCNN rawD inference 2"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_rawD_2.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "PlaneRCNN refD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_refD_0.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


echo "PlaneRCNN refD inference 1"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_refD_1.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


echo "PlaneRCNN refD inference 2"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/planercnn_refD_2.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet rawD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_rawD_0.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet rawD inference 1"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_rawD_1.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet rawD inference 2"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_rawD_2.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet refD inference 0"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_refD_0.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


echo "Mirror3DNet refD inference 1"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_refD_1.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

echo "Mirror3DNet refD inference 2"
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /project/3dlg-hcvc/mirrors/www/final_result/checkpoint/m3d_final/m3n_refD_2.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
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
--log_directory /project/3dlg-hcvc/mirrors/www/final_result/nyu_final \
--anchor_normal_npy /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy