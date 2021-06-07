
# Mirror Annotation 

## STAGE 1: Classification

- **STEP 1:**  Train a classifier 

    ```python
    python mirror3d/annotation/classifier/classifier_train.py \
    --log_directory [checkpoint and .log file saved directory] \
    --train_pos_list [training positive_sample_path.txt] \
    --train_neg_list [training negative_sample_path.txt] \
    --val_pos_list [validation positive_sample_path.txt] \
    --val_neg_list [validation negative_sample_path.txt]
    ```
    You can find Martterport3d pre-trained checkpoint for the classifier on [checkpoint.pth.tar](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/classifier_checkpoint.pth.tar)

- **STEP 2:**  Get sorted img_list with scores (saved in .json file)

    ```python
    python mirror3d/annotation/classifier/classifier_train.py \
    --unsort_img_list [img_path_to_be_sorted.txt] \
    --resume_path [classifier_checkpoint_path] \
    --output_save_folder [output_folder_path to save the output .json file]
    ```
    
- **STEP 3:**  Pick positive samples based on the .json file output by STEP 2 manually
 
    **Tip**: you can use `Mirror3D/annotation/classifier/classification_tool.py` to manually annotate mirror images
    ```python
    python mirror3d/annotation/classifier/classification_tool.py \
    --data_root [root path of the dataset] \
    --json_file_path [path of the .json file output by STEP 2] \
    --anno_output_folder [annotation result output folder] 
    ```


## STAGE 2: Mirror Mask Annotation 

We use [cvat](https://github.com/dommorin/cvat) to annotate mirror masks manually. Please refer to [cvat user guide](https://github.com/dommorin/cvat/blob/master/cvat/apps/documentation/user_guide.md) for guidance on how to do mask annotation. 

## STAGE 3: Plane Annotation
    

We provide a simple plane annotation tool `annotation/plane_annotation/plane_annotation_tool.py` to annotate the mirror plane.

Overall, we provide 12 functions in the annotation tool:

- `--function 1`: generate integer mask and colorful masks from the coco format JSON file output by CVAT
- `--function 2`: (optional) generate colorful mask from integer mask
- `--function 3`: (optional) update mirror plane information based on the refined depth map
- `--function 4`: set up the environment for annotation
- `--function 5`: use the annotation tool to annotate the mirror plane
- `--function 6`: update the depth at mirror region for uncorrected depth map
- `--function 7`: clamp outliers at mirror border
- `--function 8`: generate point cloud and 3D mesh plane from RGBD input and plane parameter
- `--function 9`: generate screenshots and videos under topdown view and front view for the point cloud and 3D mesh plane generated in function 8 
- `--function 10`: generate screenshots and videos under topdown view or front view for the point cloud and 3D mesh plane generated in function 8 
- `--function 11`: generate the colored depth map
- `--function 12`: generate HTML to view mirror color images, mirror colored depth images, and videos generated in function 9


Please follow the [example](#jump) below to get familiar with our plane annotation tool.


<span id="jump"></span>
# Getting Started 


This is an example for annotating an NYUv2 sample. 

- **STEP 1:**  Get 8-bit integer masks from CVAT output. We are going to  use coco format annotation result from CVAT:

    ```python
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 1 \
    --coco_json [path to the coco format JSON file dumped by CVAT] \
    --input_txt [path to the txt file] 
    ```

    You should generate a txt file. Each line of the text file should contain three components `[color image filename in coco json] [8-bit integer mask output path] [RGB mask output path]`

- **STEP 2:**  Set up annotation environment: please run the following command to set up the environment for annotation:

    ```python
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 4 \
    --overwrite \
    --border_width 25 \
    --input_txt docs/example/input_txt_example/anno_env_setup.txt
    ```

    Each line of the input txt file should include information: `[input color image path] [input depth image path] [input 8-bit integer mask path] [point cloud output folder (the output point cloud will be named by "[color image name]_idx_[instance id]")] [plane parameter JSON file output path] [folder to save the color image with mirror border mask] [focal length of the sample]` please refer to the example txt  `docs/example/input_txt_example/anno_env_setup.txt` for more detail. 


- **STEP 3:**  Manually annotate the mirror plane: please run the following command to try out the mirror plane annotation tool:

    ```python
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 5 \
    --annotation_progress_save_folder annotation/plane_annotation/example/anno_progess \
    --input_txt docs/example/input_txt_example/anno_update_plane.txt
    ```

    Each line of the input txt file should include information: `[input color image path] [input depth image path] [input 8-bit integer mask path] [instance point cloud path] [plane parameter JSON file output path] [path to the color image with mirror border mask] [focal length of this sample]`, please refer to the example txt `docs/example/input_txt_example/anno_update_plane.txt` for more detail. 


    <p align="center">
      <img src="figure/anno-tool-intro/anno-init.png" width=60%>
    </p>

    The above command in STEP 3 will open the annotation tool interface and show a point cloud. (The red points in the point cloud are the mirror reconstruction based on the original depth map, the green points are the mirror reconstruction based on the initial refined depth based on the RANSAC algorithm.) After viewing the point cloud, you will get the following options:

    ```shell
    ANNOTATION OPTION : 
    (1) t        : TRUE : initial plane parameter is correct
    (2) w        : WASTE : sample have error, can not be used (e.g., point cloud too noisy)
    (3) back n   : BACK : return n times (e.g., back 3 : give up the recent 3 annotated sample and go back)
    (4) goto n   : GOTO : goto the n the image (e.g., goto 3 : go to the third image
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
    
    <p align="center">
      <img src="figure/anno-tool-intro/anno-pick-point.png" width=60%>
    </p>

    This shows the user interface to pick 3 points to initialize the plane (option `i`). Press `shift + left click to select a point; press `shift + right-click to unselect; for more detail please refer to [Open3d instruction](http://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html).

    <p align="center">
      <img src="figure/anno-tool-intro/anno-mesh-plane.png" width=60%>
    </p>
    
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

    After adjustment, you can see the adjusted result. The yellow points are generated based on the adjusted mirror plane parameter. Input `f` to finish or input `a / i` to continue.


    <p align="center">
      <img src="figure/anno-tool-intro/anno-adjust.png" width=60%>
    </p>

- **STEP 4:**   Generate refined depth map: please run the following command to generate a refined depth map from the original depth map

    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 6 \
    --input_txt docs/example/input_txt_example/anno_get_refD.txt
    ```

    Each line of the input txt file should include the information: `[path to depth map to refine (rawD)] [input 8-bit integer mask path] [plane parameter JSON file output path] [path to save the refined depth map (refD)] [focal length of this sample]`, please refer to the example txt `docs/example/input_txt_example/anno_get_refD.txt` for more detail. 

- **STEP 5:**  (Optional) Clamp the refined depth map gained from STEP 4:

    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 7 \
    --input_txt docs/example/input_txt_example/anno_clamp_refD.txt \
    --expand_range 100 --clamp_dis 100 --border_width 25
    ```

    Each line of the input txt file should include the information: `[path to depth map to the unclamped refine (rawD)] [input 8-bit integer mask path] [plane parameter JSON file output path] [path to save the clamped refined depth map (refD)] [focal length of this sample]`, please refer to the example txt `docs/example/input_txt_example/anno_clamp_refD.txt` for more detail. 

- **STEP 6:**  Generate a video and colored depth map for verification: please run the following command to generate videos for verification. The videos contain the topdown view and front view of the point cloud. The output point cloud is generated based on the refined depth we get in STEP 4 and the source color image.

    To generate video, firstly, we need to generate the point cloud and 3D mesh plane:

    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 8 \
    --input_txt docs/example/input_txt_example/verification_gen_pcd_mesh.txt
    ```

    Each line of the input txt file should include the information: `[input color image path] [input depth image path] [input 8-bit integer mask path] [plane parameter JSON path] [folder to save the output point cloud] [folder to save the output mesh plane] [focal length of this sample]`, please refer to the example txt `docs/example/input_txt_example/verification_gen_pcd_mesh.txt` for more detail. 


    Then, we are going to generate video from topdown view and front view of the 3D geometry:

    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 9 \
    --above_height 3000 \
    --input_txt docs/example/input_txt_example/verification_gen_video.txt
    ```

    Each line of the input txt file should include the information: `[path to point cloud] [path to mesh plane] [screenshot output main folder]`, please refer to the example txt `docs/example/input_txt_example/verification_gen_video.txt` for more detail. 

    To better verify our annotation result, we also need to generate the colored refined depth map:

    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 11 \
    --input_txt docs/example/input_txt_example/gen_colored_depth.txt
    ```

    Each line of the input txt file should include the information: `[input depth image path] [colored depth map saved path]`, please refer to the example txt `docs/example/input_txt_example/gen_colored_depth.txt` for more detail. 

- **STEP 7:**  Launch webpage to view the videos: please run the following command to launch a website to view the video and colored depth map generated in STEP 6.


    ```shell
    python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
    --function 12 \
    --input_txt docs/example/input_txt_example/verification_gen_html.txt \
    --video_num_per_page 10 \
    --html_output_folder  output/html
    ```


    Each line of the input txt file should include the information: `[sample id] [input color image path] [colored depth map saved path] [front view video path] [topdown view video path]`, please refer to the example txt `docs/example/input_txt_example/verification_gen_html.txt` for more detail. 



    <p align="center">
      <img src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/veri-html.gif">
    </p>

    You can see the color image, colored refined depth map, and point clouds' videos on the verification web page. Please note down the sample id manually for reannotation.  
