# bts infer on refined sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path ./checkpoint/nyu/bts_refD \
--refined_depth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--log_directory output/nyu

# vnl infer on refined sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path ./checkpoint/nyu/vnl_refD.pth \
--refined_depth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--log_directory output/nyu


# saic infer on refined sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path ./checkpoint/nyu/saic_refD.pth \
--refined_depth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--log_directory output/nyu


# saic infer on raw sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path ./checkpoint/nyu/saic_rawD.pth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--log_directory output/nyu


# planercnn on refined sensor depth
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path ./checkpoint/mp3d/planercnn_refD.pth \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory output/nyu


# planercnn on raw sensor depth
python mirror3dnet/run_mirror3dnet.py \
--refined_depth \
--resume_checkpoint_path ./checkpoint/mp3d/planercnn_rawD.pth \
--config mirror3dnet/config/planercnn_config.yml \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory output/nyu



# mirror3dnet on refined sensor depth 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--resume_checkpoint_path ./checkpoint/mp3d/mirror3dnet_refD.pth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory output/nyu


# mirror3dnet on raw sensor depth 
python mirror3dnet/run_mirror3dnet.py \
--refined_depth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ./checkpoint/mp3d/mirror3dnet_rawD.pth \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory output/nyu

# mirror3dnet only normal 10 anchor normal
python mirror3dnet/run_mirror3dnet.py \
--refined_depth \
--resume_checkpoint_path ./checkpoint/mp3d/mirror3dnet_normal_10.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_val ../network_input_json/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--mesh_depth \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory output/nyu