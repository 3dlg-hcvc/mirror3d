# nyu txt and m3n 0
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_0.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_0.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
    --coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_focal_len 519 \
    --depth_shift 1000 \
    --input_height 480 \
    --input_width 640 \
    --batch_size 8 \
    --checkpoint_save_freq 1500 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
    --ref_mode DE_border \
    --anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
    --to_ref_txt $one_txt
done


# nyu txt and m3n 1
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_1.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_1.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
    --coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_focal_len 519 \
    --depth_shift 1000 \
    --input_height 480 \
    --input_width 640 \
    --batch_size 8 \
    --checkpoint_save_freq 1500 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
    --ref_mode DE_border \
    --anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
    --to_ref_txt $one_txt
done

# nyu txt and m3n 2
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_2.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_2.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
    --coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
    --coco_focal_len 519 \
    --depth_shift 1000 \
    --input_height 480 \
    --input_width 640 \
    --batch_size 8 \
    --checkpoint_save_freq 1500 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
    --ref_mode DE_border \
    --anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
    --to_ref_txt $one_txt
done

# nyu rawD plus Mirror3dNet random
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_normal_rawD_resume_2021-05-08-00-16-02/model_0014999.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--anchor_normal_npyf \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directoryf\
--ref_mode rawD_border


# nyu rawD plus Mirror3dNet 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_0.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
--ref_mode rawD_border


# nyu rawD plus Mirror3dNet 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_1.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
--ref_mode rawD_border

# nyu rawD plus Mirror3dNet 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_2.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
--ref_mode rawD_border