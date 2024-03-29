# bts infer on refined sensor depth
python mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/bts_refD \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder ../output/nyu

# bts infer on raw sensor depth
python mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/bts_nyu_v2_pytorch_densenet161/model \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder ../output/nyu


# vnl infer on refined sensor depth
python mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/vnl_refD.pth \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--output_save_folder ../output/nyu


# vnl infer on raw sensor depth
python mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/nyu_rawdata.pth \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--output_save_folder ../output/nyu


# saic infer on refined sensor depth
python mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/saic_refD.pth \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--output_save_folder ../output/nyu


# saic infer on raw sensor depth
python mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path ../checkpoint/nyu/saic_rawD.pth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder ../output/nyu


# planercnn on refined sensor depth
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path ../checkpoint/mp3d/planercnn_refD.pth \
--config mirror3d/mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/nyu


# planercnn on raw sensor depth
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/planercnn_rawD.pth \
--config mirror3d/mirror3dnet/config/planercnn_config.yml \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/nyu



# mirror3dnet on refined sensor depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_refD.pth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/nyu


# mirror3dnet on raw sensor depth 
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_rawD.pth \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/nyu

# mirror3dnet only normal 10 anchor normal
python mirror3d/mirror3dnet/run_mirror3dnet.py \
--eval \
--refined_depth \
--resume_checkpoint_path ../checkpoint/mp3d/mirror3dnet_normal_10.pth \
--config mirror3d/mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_val ../mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_val_root ../dataset/nyu \
--coco_focal_len 519 \
--mesh_depth \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
--log_directory ../output/nyu