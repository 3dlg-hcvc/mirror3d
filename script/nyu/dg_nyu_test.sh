# bts test on official checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path checkpoint/nyu/bts_official_nyu_raw \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output

# bts test on rawD checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path checkpoint/nyu/bts_rawD \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output


# bts test on refD checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path checkpoint/nyu/bts_refD \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output


# vnl test on official checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path checkpoint/nyu/vnl_official_nyu_raw.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output


# vnl test on rawD checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path checkpoint/nyu/vnl_rawD.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output

# vnl test on refD checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path checkpoint/nyu/vnl_refD.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output


# saic test on rawD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path checkpoint/nyu/saic_rawD.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output

# saic test on refD checkpoint
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path checkpoint/nyu/saic_refD.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output







