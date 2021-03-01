# bts train on refined depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_val_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--learning_rate 1e-5 \
--log_directory output

# bts train on sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_val_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--learning_rate 1e-5 \
--log_directory output

# bts train on refined depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_val_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--learning_rate 1e-5 \
--log_directory output


# bts train on sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val dataset/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_val_root /local-scratch/jiaqit/exp/Mirror3D/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--learning_rate 1e-5 \
--log_directory output


