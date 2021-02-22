python tools/train_matterport.py \
--snapshot_period 10 \
--depth_shift 1000 \
--input_width 160 \
--input_height 120 \
--train_coco_path /local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_train_normalFormat_10_normal.json \
--val_coco_path /local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json \
--coco_root /local-scratch/share_data/mirror3D/nyu/nyu_crop_456_608 \
--refined_depth True


python tools/train_matterport.py \
--snapshot_period 10 \
--depth_shift 1000 \
--input_width 160 \
--input_height 120 \
--train_coco_path /local-scratch/jiaqit/exp/data/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_train_normalFormat_10_normal.json \
--val_coco_path /local-scratch/jiaqit/exp/data/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json \
--coco_root /local-scratch/jiaqit/exp/data/nyu_crop_456_608 \
--refined_depth True


python tools/train_matterport.py \
--snapshot_period 10 \
--depth_shift 1000 \
--input_width 160 \
--input_height 120 \
--train_coco_path /local-scratch/jiaqit/exp/data/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_train_normalFormat_10_normal.json \
--val_coco_path /local-scratch/jiaqit/exp/data/nyu_crop_456_608/coco_input/with_neg_1280_1024/pos_test_normalFormat_10_normal.json \
--coco_root /local-scratch/jiaqit/exp/data/nyu_crop_456_608


python tools/train_matterport.py \
--snapshot_period 50 \
--depth_shift 4000 \
--epoch 200 \
--input_width 160 \
--input_height 128 \
--train_batch_size 32 \
--train_coco_path /home/jiaqit/scratch/data/coco_input/debug/pos_train_only_DE.json \
--val_coco_path /home/jiaqit/scratch/data/coco_input/debug/pos_val_only_DE.json \
--coco_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/m3d_unzip \
--refined_depth False



python tools/train_matterport.py \
--snapshot_period 50 \
--depth_shift 4000 \
--epoch 200 \
--input_width 160 \
--input_height 128 \
--train_batch_size 8 \
--train_coco_path /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip/coco_input/DE_1280_1024_all/pos_train_only_DE.json \
--val_coco_path /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip/coco_input/DE_1280_1024_all/pos_val_only_DE.json \
--coco_root /project/3dlg-hcvc/jiaqit/m3d/m3d_unzip