# planercnn on refD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/planercnn_config.yml \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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


# planercnn on rawD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/planercnn_config.yml \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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


# mirror3dnet on refD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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


# mirror3dnet on rawD 
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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



# mirror3dnet (only normal)
python mirror3dnet/run_mirror3dnet.py \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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



# mirror3dnet on refD  based on m3d pretrained
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path checkpoint/m3d_pretrained_m3n.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--refined_depth \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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


# mirror3dnet on rawD  based on m3d pretrained
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path checkpoint/m3d_pretrained_m3n.pth \
--config mirror3dnet/config/mirror3dnet_config.yml \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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



# mirror3dnet (only normal) based on m3d pretrained
python mirror3dnet/run_mirror3dnet.py \
--resume_checkpoint_path checkpoint/m3d_pretrained_m3n.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
--coco_train dataset/nyu/with_mirror/precise/network_input_json/train_10_normal_mirror.json \
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