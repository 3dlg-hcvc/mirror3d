# bts test on refD checkpoint 0
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_0 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# bts test on refD checkpoint 1
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_1 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# bts test on refD checkpoint 2
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_2 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# vnl test on refD checkpoint 0
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# vnl test on refD checkpoint 1
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# vnl test on refD checkpoint 2
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# saic test on rawD checkpoint 0
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# saic test on rawD checkpoint 1
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# saic test on rawD checkpoint 2
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# saic test on refD checkpoint 0
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# saic test on refD checkpoint 1
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# saic test on refD checkpoint 2
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# bts test on official checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path /project/6049211/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_official_nyu_raw \
--coco_train /precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# vnl test on official checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path /project/6049211/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_official_nyu_raw.pth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# complete bts test on refD checkpoint 0
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_0 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete bts test on refD checkpoint 1
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_1 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete bts test on refD checkpoint 2
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_refD_2 \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# complete vnl test on refD checkpoint 0
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete vnl test on refD checkpoint 1
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete vnl test on refD checkpoint 2
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_refD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# complete saic test on rawD checkpoint 0
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete saic test on rawD checkpoint 1
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete saic test on rawD checkpoint 2
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_rawD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete saic test on refD checkpoint 0
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_0.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# complete saic test on refD checkpoint 1
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_1.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete saic test on refD checkpoint 2
python init_depth_generator/saic_depth_completion/init_depth_gen_infer.py \
--refined_depth \
--resume_checkpoint_path /home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D/checkpoint/nyu_final/saic_refD_2.pth \
--coco_train  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root  /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final


# complete bts test on official checkpoint
python init_depth_generator/bts/pytorch/init_depth_gen_infer.py \
--resume_checkpoint_path /project/6049211/jiaqit/exp/Mirror3D/checkpoint/nyu_final/bts_official_nyu_raw \
--coco_train /precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final

# complete vnl test on official checkpoint
python init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_infer.py \
--resume_checkpoint_path /project/6049211/jiaqit/exp/Mirror3D/checkpoint/nyu_final/vnl_official_nyu_raw.pth \
--coco_train /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/train_10_normal_all.json \
--coco_val /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_all.json \
--coco_train_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_val_root /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--output_save_folder /home/jiaqit/projects/rrg-msavva/jiaqit/data/Mirror3D_final/output/final_result/nyu_final