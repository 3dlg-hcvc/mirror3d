# bts train on refined depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu

# bts train on sensor depth
python init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu

# vnl train on refined depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu


# vnl train on sensor depth
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu

# saic train on refined depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--refined_depth \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu


# saic train on sensor depth
python init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--coco_train /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /project/6049211/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /project/6049211/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100 \
--learning_rate 1e-4 \
--log_directory debug/nyu


