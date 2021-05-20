# nyu_complete nyu txt and m3n 0
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_0.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_0.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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


# nyu_complete nyu txt and m3n 1
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_1.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_1.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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

# nyu_complete nyu txt and m3n 2
to_ref_txt_list="script/cedar_submission/nyu_infer/nyu_dg_2.txt"
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_2.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
    --refined_depth \
    --coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
    --coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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

# nyu_complete nyu rawD plus Mirror3dNet random
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_normal_rawD_resume_2021-05-08-00-16-02/model_0014999.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
--ref_mode rawD_border


# nyu_complete nyu rawD plus Mirror3dNet 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_0.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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


# nyu_complete nyu rawD plus Mirror3dNet 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_1.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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

# nyu_complete nyu rawD plus Mirror3dNet 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/m3d_final/m3n_normal_10_2.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
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





# nyu_complete PlaneRCNN rawD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/planercnn_rawD_resume_2021-05-16-09-46-44/model_0028499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete PlaneRCNN rawD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d1/planercnn_rawD_resume_2021-05-15-10-48-10/model_0023999.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete PlaneRCNN rawD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/planercnn_rawD_resume_2021-05-14-09-37-30/model_0013499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# nyu_complete PlaneRCNN refD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/planercnn_refD_resume_2021-05-15-02-29-08/model_0029999.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete PlaneRCNN refD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d1/planercnn_refD_resume_2021-05-15-03-58-41/model_0025499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete PlaneRCNN refD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/planercnn_refD_resume_2021-05-15-04-15-41/model_0025499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete Mirror3DNet rawD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_rawD_resume_2021-05-08-20-37-26/model_0016499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# nyu_complete Mirror3DNet rawD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/m3n_full_rawD_resume_2021-05-15-03-51-08/model_0037499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete Mirror3DNet rawD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/m3n_full_rawD_resume_2021-05-15-04-11-22/model_0037499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete Mirror3DNet refD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-23-13-10/model_0014999.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# nyu_complete Mirror3DNet refD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-23-00-37/model_0022499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# nyu_complete Mirror3DNet refD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-20-37-26/model_0019499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy



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
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final \
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





# PlaneRCNN rawD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/planercnn_rawD_resume_2021-05-16-09-46-44/model_0028499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# PlaneRCNN rawD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d1/planercnn_rawD_resume_2021-05-15-10-48-10/model_0023999.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# PlaneRCNN rawD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/planercnn_rawD_resume_2021-05-14-09-37-30/model_0013499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# PlaneRCNN refD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/planercnn_refD_resume_2021-05-15-02-29-08/model_0029999.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# PlaneRCNN refD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d1/planercnn_refD_resume_2021-05-15-03-58-41/model_0025499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# PlaneRCNN refD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/planercnn_refD_resume_2021-05-15-04-15-41/model_0025499.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# Mirror3DNet rawD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_rawD_resume_2021-05-08-20-37-26/model_0016499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# Mirror3DNet rawD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d0/m3n_full_rawD_resume_2021-05-15-03-51-08/model_0037499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# Mirror3DNet rawD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3d2/m3n_full_rawD_resume_2021-05-15-04-11-22/model_0037499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# Mirror3DNet refD inference 0
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-23-13-10/model_0014999.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final0 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy


# Mirror3DNet refD inference 1
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-23-00-37/model_0022499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final1 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

# Mirror3DNet refD inference 2
python mirror3dnet/run_mirror3dnet.py \
--eval \
--mesh_depth \
--eval_save_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/debug/m3n_full_refD_resume_2021-05-08-20-37-26/model_0019499.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 1 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/output/final_result/nyu_final/dt_final2 \
--anchor_normal_npy /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/mirror_normal/m3d/m3d_kmeans_normal_10.npy

