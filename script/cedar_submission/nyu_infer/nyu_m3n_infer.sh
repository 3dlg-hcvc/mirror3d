# txt and m3n 0
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


# txt and m3n 1
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

# txt and m3n 2
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
