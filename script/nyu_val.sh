# bts test on refined depth
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/bts_nyu_v2_pytorch_resnet50/model \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output


# vnl test on refined depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path /local-scratch/jiaqit/exp/Mirror3D/checkpoint/vnl_official_nyu_raw.pth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder output



# vnl test on sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output

# saic test on refined depth
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output


# saic test on sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root dataset/nyu \
--coco_val_root dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory output


