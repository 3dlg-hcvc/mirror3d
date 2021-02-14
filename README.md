# Mirror3D

## Enviroment Setup

python 3.7.4

```shell
export PYTHONPATH=[Mirror3D repository absolute path]

```

## Annotation Tool

### Classification

- STEP 1 : Train a classifier 

```python
python Mirror3D/annotation/classifier/classifier_train.py --log_directory [checkpoint and .log file saved directory] --train_pos_list [training positive_sample_path.txt] --train_neg_list [training negative_sample_path.txt] --val_pos_list [validation positive_sample_path.txt] --val_neg_list [validation negative_sample_path.txt]

```
Pretrained classifier's checkpoint can be found on [google drive](https://www.example.com)

- STEP 2 : Get sorted img_list with score 

```python
python Mirror3D/annotation/classifier/classifier_train.py --unsort_img_list [img_path_to_be_sorted.txt] --resume_path [classifier_checkpoint_path] --output_save_folder [output_folder_path to save the output txt]

```

- STEP 3 : Use `Mirror3D/annotation/classifier/classification_tool.py` to manully annoatate mirror images


```python
python Mirror3D/annotation/classifier/classification_tool.py --folder [folder contains images] --json_file_path [json file output by STEP 2] --labels [label you want to have for the input images, e.g. "mirror", "no mirror"] --exclusion [path list .txt which you want to exclude]  --output_file_path [.txt file path to store the annotation result]

```

### Plane annoatation

```python
python Mirror3D/annotation/plane_annotation_tool/plane_annotation_tool.py --stage [all / 1 ~ 6] --data_main_folder [dataset main folder] --process_index [the process index during multi-processing] --border_width [mirror border width] --f [focal length of the dataset] --anno_output_folder [annotation result output folder]
```

- `--stage 1` : Set up annotation environment 

- `--stage 2` : Manully annotate the mirror plane

- `--stage 3` : Update raw depth

- `--stage 4` : Clamp depth data (clamp outlier around the mirror instance)

- Other post processing function 
	- `--stage 5` update img_info based on refined depth

- `--stage all` : Run stage 1 ~ 4 together

- Note : 
	- Only changing the ratio of w:h (and don’t change f)  will change the point cloud’s shape
	- Only changing the depth_shift (all depth change in the same way) OR changing img_shape but keep w:h will not change the shape of the point cloud (but the point cloud would be small / larger / closer / further)
	- During training: If the image is changed by ration r (w:h ratio doesn’t change), f should also change by ratio r



### Verification

(TODO)