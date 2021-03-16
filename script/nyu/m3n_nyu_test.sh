# other depth + Mirror3dNet
echo "Enter the output file list.txt to refine: "  
read to_ref_txt_list
for one_txt in $(cat $to_ref_txt_list); do
    echo "applied Mirror3dNet on $one_txt"
    python mirror3dnet/run_mirror3dnet.py \
    --eval \
    --resume_checkpoint_path checkpoint/model_0020999.pth \
    --config mirror3dnet/config/mirror3dnet_normal_config.yml \
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
    --log_directory output \
    --ref_mode DE_border \
    --to_ref_txt $one_txt
done

# rawD + Mirror3dNet
python mirror3dnet/run_mirror3dnet.py \
--eval \
--resume_checkpoint_path checkpoint/model_0020999.pth \
--config mirror3dnet/config/mirror3dnet_normal_config.yml \
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
--log_directory output \
--ref_mode rawD_border

