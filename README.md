# Mirror3D

## Environment Setup

python 3.7.4

```shell
export PYTHONPATH=[Mirror3D repository absolute path]
```

## Annotation Tool

### Classification

- STEP 1: Train a classifier 

	```python
	python Mirror3D/annotation/classifier/classifier_train.py --log_directory [checkpoint and .log file saved directory] --train_pos_list [training positive_sample_path.txt] --train_neg_list [training negative_sample_path.txt] --val_pos_list [validation positive_sample_path.txt] --val_neg_list [validation negative_sample_path.txt]
	```
Reference pretrained classifier's checkpoint can be found on [checkpoint.pth.tar](http://aspis.cmpt.sfu.ca/projects/mirrors/checkpoint/classifier_checkpoint/checkpoint.pth.tar)

- STEP 2: Get sorted img_list with scores

	```python
	python Mirror3D/annotation/classifier/classifier_train.py --unsort_img_list [img_path_to_be_sorted.txt] --resume_path [classifier_checkpoint_path] --output_save_folder [output_folder_path to save the output txt]
	```

- STEP 3: Use `Mirror3D/annotation/classifier/classification_tool.py` to manually annotate mirror images


	```python
	python Mirror3D/annotation/classifier/classification_tool.py --folder [folder contains images] --json_file_path [json file output by STEP 2] --labels [label you want to have for the input images, e.g. "mirror", "no mirror"] --exclusion [path list .txt which you want to exclude]  --output_file_path [.txt file path to store the annotation result]
	```

### Mirror mask annotation 

We use [cvat](https://github.com/dommorin/cvat) to annotate mirror mask manully. About how to annotate object's mask, please refer to [cvat user guide](https://github.com/dommorin/cvat/blob/master/cvat/apps/documentation/user_guide.md).
### Plane annoatation

```python
python Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py --stage [all / 1 ~ 6] --data_main_folder [dataset main folder] --process_index [the process index during multi-processing] --border_width [mirror border width] --f [focal length of the dataset] --anno_output_folder [annotation result output folder]
```

- `--stage 1`: Set up annotation environment 

- `--stage 2`: Manually annotate the mirror plane

- `--stage 3`: Update raw depth

- `--stage 4`: Clamp depth data (clamp outlier around the mirror instance)

- Other post processing function 
	- `--stage 5` update img_info based on refined depth

- `--stage all` : Run stage 1 ~ 4 together

- Note : 
	- Only changing the ratio of w:h (and don't change f)  will change the point cloud's shape
	- Only changing the depth image OR resizing the image but keep the w:h ratio will not change the shape of the point cloud (but the point cloud would be closer / further/ small / larger )
	- During training: If the image is resized by ratio x (w:h ratio doesn't change), f should also multiply ratio x


### Verification

- STEP 1: Generate video for vrification 
	```python
	python Mirror3D/visualization/dataset_visualization.py --stage all --data_main_folder [dataset main folder] --process_index [the process index during multi-processing]  --multi_processing --overwrite --f [focal length of the dataset] --output_folder [output point cloud/ mesh plane/ screenshot/ video saved folder] --view_mode [topdown/ front]
	```

	- `--stage 1`: Generate point cloud and mesh plane for visualization

	- `--stage 2`: Generate screenshot for "point cloud + mesh plane" under "topdown + front" view

	- `--stage 3`: Generate screenshot for "point cloud + mesh plane" under specific view
	- `--stage 4`: Generate videos under "topdown + front" view
	- `--stage 5`: Generate videos under specific view
	- `--stage all`: run stage 1, 2, 4 together

- STEP 2: Launch webpage to view the videos
	
	```python 
	Mirror3D/annotation/verification/verification.py --stage 1 --data_main_folder [folder that contains "video_front, video_topdown .. etc" folders] --output_folder [.html files output folder] --video_num_per_page [int: how many video to display in one .html]
	```

	Annotators should manully note down the error sample's path to a [error_sample].txt

- STEP 3: Copy out the error sample's data to another folder for reannotation

	```python 
	Mirror3D/annotation/verification/verification.py --stage 2 --data_main_folder [dataset main folder] --output_folder [folder to save the copy of data] --error_list [.txt that contains the error samples' name]
	```