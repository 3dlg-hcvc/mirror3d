# Mirror3D

## Environment Setup

- python 3.7.4
- [Detectron2](https://github.com/facebookresearch/detectron2): `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
- `pip install -r requirements.txt`
- Pull submodules `git submodule update --init --recursive`



## Preparation for all implementations

```shell
$ mkdir workspace
$ cd workspace
### Put data under dataset folder
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/3dlg-hcvc/Mirror3D.git

```

## Dataset

Please refer to [Mirror3D Dataset](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/Mirror3D_dataset.md) for instruction on preparing mirror data.

## Models

### Mirror3DNet PyTorch Implementation

Mirror3DNet architecture can be used for either an RGB image or an RGBD image input. For an RGB input, we refine the depth of the predicted depth map D<sub>pred</sub> output by a depth estimation module. For RGBD input, we refine a noisy input depth D<sub>noisy</sub>.
![network-arch](docs/figure/network-arch-cr-new.png)

Please check [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet) for our network pytorch implementation. 

### Initial Depth Generator Implementation

We test three methods on our dataset:

- [BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cogaplex-bts/bts)
- [VNL: Enforcing geometric constraints of virtual normal for depth prediction](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)
- [saic : Decoder Modulation for Indoor Depth Completion](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37)

We have updated the dataloader and the main train/test script to support our input format. 

## Model Zoo

| Source Dataset | Input | Train | Method                                                                                                  | Model Download |
|----------------|-------|-------|---------------------------------------------------------------------------------------------------------|----------------|
| NYUv2          | RGBD  |       | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) |                |
| NYUv2          | RGBD  |       |                                                                                                         |                |
| NYUv2          |       |       |                                                                                                         |                |
|                |       |       |                                                                                                         |                |


## Training
To train initial depth generator:

```shell
### Train on NYUv2 mirror data
bash script/nyu_train.sh
### Train on Matterport3D mirror data
bash script/mp3d_train.sh
```

## Inference
To test the three initial depth generator:

```shell
### Run the inferece on NYUv2 mirror data
script/nyu_infer.sh
### Run the inferece on Matterport3D mirror data
script/mp3d_infer.sh
```

## Annotation Tool

### Classification

- STEP 1: Train a classifier 

    ```python
    python annotation/classifier/classifier_train.py --log_directory [checkpoint and .log file saved directory] --train_pos_list [training positive_sample_path.txt] --train_neg_list [training negative_sample_path.txt] --val_pos_list [validation positive_sample_path.txt] --val_neg_list [validation negative_sample_path.txt]
    ```
You can find Martterport3d pre-trained checkpoint for the classifier on [checkpoint.pth.tar](http://aspis.cmpt.sfu.ca/projects/mirrors/checkpoint/classifier_checkpoint/checkpoint.pth.tar)

- STEP 2: Get sorted img_list with scores (saved in .json file)

    ```python
    python annotation/classifier/classifier_train.py --unsort_img_list [img_path_to_be_sorted.txt] --resume_path [classifier_checkpoint_path] --output_save_folder [output_folder_path to save the output .json file]
    ```
    
- STEP 3: Pick positive samples based on the .json file output by STEP 2 manually

<!--
- STEP 3: Use `Mirror3D/annotation/classifier/classification_tool.py` to manually annotate mirror images


    ```python
    python annotation/classifier/classification_tool.py --data_root [root path of the dataset] --json_file_path [path of the .json file output by STEP 2] --anno_output_folder [annotation result output folder] 
    ```
-->

### Mirror mask annotation 

We use [cvat](https://github.com/dommorin/cvat) to annotate mirror mask manually. Please refer to [cvat user guide](https://github.com/dommorin/cvat/blob/master/cvat/apps/documentation/user_guide.md) for guidance on mask annotation. 
### Plane annoatation

```python
python annotation/plane_annotation/plane_annotation_tool.py --stage [all / 1 ~ 6] \
--data_main_folder [dataset main folder] \
--multi_processing [optional] --process_index [optional: the process index during multi-processing] \
--border_width [mirror border width] \
--f [focal length of the dataset] \
--overwrite [optional : for --stage 1, overwrite the output result or not] \
--mask_version [mirror mask version: precise (default) / coarse] \
--anno_output_folder [optional : annotation result output folder; save annotation result under --data_main_folder by default ]
```

- `--stage 1`: Set up annotation environment (You should have folders contain the mirror mask, mirror RGBD under `--data_main_folder`. You should store mirror color images under a folder named `mirror_color_images`; You should store mirror raw depth images under a folder named `raw_meshD` or `raw_sensorD`; mirror masks images should be stored under a folder named `mirror_instance_mask_precise` or `mirror_instance_mask_coarse`)

- `--stage 2`: Manually annotate the mirror plane based on our plane annotation tool, check [User Instruction](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/user_instruction.md) for how to use the plane annotation tool.

- `--stage 3`: Get refined depth map from original depth map

- `--stage 4`: Clamp depth map (clamp outlier around the mirror instance)

- Other post processing function 
    - `--stage 5` update img_info based on refined depth

- `--stage all` : Run stage 1 ~ 4 together

- Note : 
    - Only changing the ratio of w:h (and do not change f)  will change the pointcloud's shape
    - Only changing the depth image OR resizing the image but keep the w:h ratio will not change the shape of the point cloud (but the point cloud would be closer / further/ small / larger )
    - During training: If the image is resized by ratio x (w:h ratio does not change), f should also multiply ratio x
    - the --data_main_folder need to contain "scannet" if you are annotating Scannet dataset; "mp3d" for Matterport3d dataset; "nyu" for NYUv2 dataset; only .png image is supported; Apart from Matterpot3d other dataset's color image name and depth image name should be the same. 

### Verification

- STEP 1: Generate video for verification 
    ```python
    python visualization/dataset_visualization.py --stage all --data_main_folder [dataset main folder] --process_index [the process index during multi-processing]  --multi_processing --overwrite --f [focal length of the dataset] --output_folder [output point cloud/ mesh plane/ screenshot/ video saved folder] --view_mode [topdown/ front]
    ```

    - `--stage 1`: Generate point cloud and mesh plane for visualization

    - `--stage 2`: Generate screenshot for "point cloud + mesh plane" under "topdown + front" view

    - `--stage 3`: Generate screenshot for "point cloud + mesh plane" under specific view
    - `--stage 4`: Generate videos under "topdown + front" view
    - `--stage 5`: Generate videos under specific view
    - `--stage 6`: Generate refined depth colorful heatmap
    - `--stage 7`: Generate relevant data information (mirror ratio, mirror area max depth, etc)
    - `--stage 8`: Generate data distribution figures
    - `--stage all`: run stage 1, 2, 4, 6 together

- STEP 2: Launch webpage to view the videos
    
    ```python 
    python annotation/plane_annotation/verification.py --stage 1 --data_main_folder [folder that contains "video_front, video_topdown .. etc" folders] --output_folder [.html files output folder] --video_num_per_page [int: how many video to display in one .html]
    ```

    Annotators should manually note down the error sample's path to a [error_sample].txt

- STEP 3: Copy/ move out the error sample's data to another folder for reannotation

    ```python 
    python annotation/plane_annotation/verification.py --stage 2 --data_main_folder [dataset main folder] --output_folder [folder to save the copy of data] --error_list [.txt that contains the error samples' name] --move [bool, if ture it will move out all the error samples' information, if false, it will copy out all the error samples' information]
    ```
