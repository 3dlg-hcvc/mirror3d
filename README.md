# Mirror3D

## Enviroment Setup

python 3.7.4

```shell
export PYTHONPATH=[Mirror3D repository absolute path]

```

## Annotation Tool

### Classification

- Train a classifier 

```python
python Mirror3D/annotation/classifier/classifier_train.py --log_directory [checkpoint and .log file saved directory] --train_pos_list [training positive_sample_path.txt] --train_neg_list [training negative_sample_path.txt] --val_pos_list [validation positive_sample_path.txt] --val_neg_list [validation negative_sample_path.txt]

```
Pretrained classifier's checkpoint can be found on [google drive](https://www.example.com)

- Get sorted img_list with score 

```python
python Mirror3D/annotation/classifier/classifier_train.py --unsort_img_list [img_path_to_be_sorted.txt] --resume_path [classifier_checkpoint_path] --output_save_folder [output_folder_path to save the output txt]

```

- Use ** to manully annoatate mirror images


### Plane annoatation


### Verification