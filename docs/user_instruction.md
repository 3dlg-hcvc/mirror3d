
# Mirror Annotation 

## STAGE 1: Classification

- **STEP 1** Train a classifier 

    ```python
    python annotation/classifier/classifier_train.py \
    --log_directory [checkpoint and .log file saved directory] \
    --train_pos_list [training positive_sample_path.txt] \
    --train_neg_list [training negative_sample_path.txt] \
    --val_pos_list [validation positive_sample_path.txt] \
    --val_neg_list [validation negative_sample_path.txt]
    ```
You can find Martterport3d pre-trained checkpoint for the classifier on [checkpoint.pth.tar](http://aspis.cmpt.sfu.ca/projects/mirrors/checkpoint/classifier_checkpoint/checkpoint.pth.tar)

- **STEP 2** Get sorted img_list with scores (saved in .json file)

    ```python
    python annotation/classifier/classifier_train.py \
    --unsort_img_list [img_path_to_be_sorted.txt] \
    --resume_path [classifier_checkpoint_path] \
    --output_save_folder [output_folder_path to save the output .json file]
    ```
    
- **STEP 3** Pick positive samples based on the .json file output by STEP 2 manually
 
    **Tip**: you can use `Mirror3D/annotation/classifier/classification_tool.py` to manually annotate mirror images
    ```python
    python annotation/classifier/classification_tool.py \
    --data_root [root path of the dataset] \
    --json_file_path [path of the .json file output by STEP 2] \
    --anno_output_folder [annotation result output folder] 
    ```


## STAGE 2: Mirror Mask Annotation 

We use [cvat](https://github.com/dommorin/cvat) to annotate mirror mask manually. Please refer to [cvat user guide](https://github.com/dommorin/cvat/blob/master/cvat/apps/documentation/user_guide.md) for guidance on mask annotation. 
## STAGE 3: Plane Annoatation

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

## STAGE 4: Annotation Verification

- **STEP 1** Generate video for verification 
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

- **STEP 2** Launch webpage to view the videos
    
    ```python 
    python annotation/plane_annotation/verification.py \
    --stage 1 \
    --data_main_folder [folder that contains "video_front, video_topdown .. etc" folders] \
    --output_folder [.html files output folder] \
    --video_num_per_page [int: how many video to display in one .html]
    ```

    Annotators should manually note down the error sample's path to a [error_sample].txt

- **STEP 3** Copy/ move out the error sample's data to another folder for reannotation

    ```python 
    python annotation/plane_annotation/verification.py \
    --stage 2 \
    --data_main_folder [dataset main folder] \
    --output_folder [folder to save the copy of data] \
    --error_list [.txt that contains the error samples' name] --move [bool, if ture it will move out all the error samples' information, if false, it will copy out all the error samples' information]
    ```



## Getting Started

Here is quick example, we are going to annotate several samples from NYUv2 and get the refined depth map based on the precise mask:

- **STEP 1** Prepare data structure: after getting the annotated mirror mask, we store the original data in the following format:

```shell
├── mirror_color_images
│   ├── 221.jpg
│   ├── 45.jpg
│   └── 686.jpg
├── mirror_instance_mask_precise
│   ├── 221.png
│   ├── 45.png
│   └── 686.png
└── raw_sensorD
    ├── 221.png
    ├── 45.png
    └── 686.png
```

- **STEP 2** Set up annotation environment: please run the following command to set up the environment:

```python
python annotation/plane_annotation/plane_annotation_tool.py --stage 1 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```

Then you will get the output pointclouds, RANSAC initialized mirror plane parameter information, and masked image for each mirror instance. The output should be like this:

```shell
├── anno_pcd
│   ├── 221_idx_008000.ply # We set the hexadecimal of the instances' masks' RBD value as instances' id. 
│   ├── 221_idx_800000.ply
│   ├── 45_idx_800000.ply
│   ├── 686_idx_008000.ply
│   └── 686_idx_800000.ply
├── border_vis # color images with visualized mirror mask
│   ├── 221_idx_008000.jpg
│   ├── 221_idx_800000.jpg
│   ├── 45_idx_800000.jpg
│   ├── 686_idx_008000.jpg
│   └── 686_idx_800000.jpg
└── mirror_plane
    ├── 221.json
    ├── 45.json
    └── 686.json
```



- **STEP 3** Manually annotate the mirror plane: please run the following command to use the mirror plane annotation tool:

```python
python annotation/plane_annotation/plane_annotation_tool.py --stage 2 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```
![pick point pic](figure/anno-tool-intro/anno-init.png)
This command will open the annotation tool interface and show a pointcloud. (The red points in the pointcloud are the mirror reconstruction based on the orginal depth map, the green points are the mirror reconstruction based on the initial refined depth based on RANSAC algorithm.) After viewing the pointcloud you will get the following options:

```shell
ANNOTATION OPTION : 
(1) t        : TRUE : initial plane parameter is correct
(2) w        : WASTE : sample have error, can not be used (e.g. point cloud too noisy)
(3) back n   : BACK : return n times (e.g., back 3 : give up the recent 3 annotated sample and go back)
(4) goto n   : GOTO : goto the n th image (e.g., goto 3 : go to the third image
(5) n        : NEXT : goto next image without annotation
(6) a        : ADJUST: adjust one sample repeatedly
(7) exit     : EXIT : save and exit
```


If you want to adjust the mirror plane, please input option `a' (ADJUST). 

'a ADJUST` has the following options:


```shell
ADJUST ONE SAMPLE OPTION : 
(1) f        : FINISH : update refined_sensorD/ refined_meshD/ img_info and EXIT
(2) a        : ADJUST : adjust the plane parameter based on the current plane parameter
(3) i        : INIT : pick 3 points to initialize the plane
```
![pick point pic](figure/anno-tool-intro/anno-pick-point.png)

This shows the user interface to pick 3 points to initialize the plane (option `i`). Press `shift + left click` to select a point; press `shift + right click` to unselect; for more detail please refer to [Open3d instruction](http://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html).

![blue_plane](figure/anno-tool-intro/anno-mesh-plane.png)
This shows the user interface to adjust the plane parameter based on the current plane parameter (option `a'). To adjust the light blue plane, please follow:

```shell
ADJUST ONE PLANE OPTION : 
(1) a        : plane move left
(2) w        : plane move up
(3) s        : plane move down
(4) d        : plane move right
(5) e        : plane move closer
(6) r        : plane move further
(7) i        : make the plane larger
(8) k        : make the plane smaller
(9) j        : rotate left
(10) l       : rotate right
(11) o       : rotate upwards
(12) p       : rotate downwards
```

After adjustment, you can see the adjusted result. The yellow points are generated based on the adjusted mirror plane paramenter. Input `f` to finish or input `a / i` to continue.
![adjust-view](figure/anno-tool-intro/anno-adjust.png)


- **STEP 4**  Generate refined depth map: please run the following command to generate a refined depth map from the original depth map

```shell
python annotation/plane_annotation/plane_annotation_tool.py --stage 3 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```

- **STEP 5** Generate video for verification: please run the following command to generate videos for verification. The videos contain the topdown view and front view of the refined pointcloud. The output refined pointcloud is generated based on the refined depth we get in STEP 4 and the source color image.

```shell
python visualization/dataset_visualization.py --stage all \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```


- **STEP 6** Launch webpage to view the videos: please run the following command to launch a website to view the video generated in STEP 5.

```shell
python annotation/plane_annotation/verification.py \
--stage 1 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--video_main_folder [folder that contains "video_front, video_topdown .. etc" folders] \
--output_folder docs/figure/anno-tool-example/nyu/html \
--video_num_per_page 10

```

- **STEP 7** Copy/ move out the error sample's data to another folder for reannotation. 


```shell
python annotation/plane_annotation/verification.py \
--stage 2 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--output_folder docs/figure/anno-tool-example/nyu_reannoatate \
--error_list [.txt that contains the error samples' name (without file name extension); error sample is the sample with the wrong annotation] \
--waste_list [.txt that contains the waste samples' name (without file name extension); the waste sample is the sample to waste] \
--move [bool, if true it will move out all the error samples' information, if false, it will copy out all the error samples' information]
```
