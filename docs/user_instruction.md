
### User instruction

This is a detailed instruction for our plane annotation tool. Here, we are going to annotate several samples from NYUv2 and get the refined depth map based on the precise mask:

-STEP 1: Prepare data structure: after getting the annotated mirror mask, we store the original data in the following format:

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

- STEP 2: Set up annotation environment: please run the following command to set up the environment:

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



- STEP 3: Manually annotate the mirror plane: please run the following command to use the mirror plane annotation tool:

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

This shows the user interface to pick 3 points to initialize the plane (option `i'). Press `shift + left click` to select a point; press `shift + right click` to unselect; for more detail please refer to [Open3d instruction](http://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html).

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


- STEP 4:  Generate refined depth map: please run the following command to generate a refined depth map from the original depth map

```shell
python annotation/plane_annotation/plane_annotation_tool.py --stage 3 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```

- STEP 5: Generate video for verification: please run the following command to generate videos for verification. The videos contain the topdown view and front view of the refined pointcloud. The output refined pointcloud is generated based on the refined depth we get in STEP 4 and the source color image.

```shell
python visualization/dataset_visualization.py --stage all \
--data_main_folder docs/figure/anno-tool-example/nyu \
--f 519
```


-- STEP 6: Launch webpage to view the videos: please run the following command to launch a website to view the video generated in STEP 5.

```shell
python annotation/plane_annotation/verification.py \
--stage 1 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--video_main_folder [folder that contains "video_front, video_topdown .. etc" folders] \
--output_folder docs/figure/anno-tool-example/nyu/html \
--video_num_per_page 10

```

-- STEP 7: Copy/ move out the error sample's data to another folder for reannotation. 


```shell
python annotation/plane_annotation/verification.py \
--stage 2 \
--data_main_folder docs/figure/anno-tool-example/nyu \
--output_folder docs/figure/anno-tool-example/nyu_reannoatate \
--error_list [.txt that contains the error samples' name (without file name extension); error sample is the sample with the wrong annotation] \
--waste_list [.txt that contains the waste samples' name (without file name extension); the waste sample is the sample to waste] \
--move [bool, if true it will move out all the error samples' information, if false, it will copy out all the error samples' information]
```
