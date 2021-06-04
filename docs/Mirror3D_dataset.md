# Mirror3D 

Mirror3D is a large-scale 3D mirror plane annotation dataset based on three popular RGBD datasets ([Matterpot3D](https://niessner.github.io/Matterport/),[NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [ScanNet](http://www.scan-net.org/)) containing 7,011 mirror instance masks and 3D planes.

Please visit our [project website]() for updates and to browse the data.

## Download

- [mp3d.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/mp3d.zip)
- [scannet.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/scannet.zip)
- [nyu.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/nyu.zip)

## Mirror3D Data Organization

The unzipped mirror data we provided are stored in the following structures:


### NYUv2-small (nyu.zip) / ScanNet (scannet.zip) / Matterport3d (mp3d.zip)


```shell
nyu/scannet/mp3d
├── mirror_instance_mask_coarse # 8-bit coarse instance-level mirror segmentation mask
└── mirror_instance_mask_precise # 8-bit precise instance-level mirror segmentation mask
└── delta_depth_coarse # delta image to generate the coarse refined depth map
└── delta_depth_precise # delta image to generate the precise refined depth map
└── mirror_plane # mirror plane parameter information 
```

- For NYUv2 mirror data is stored as [data type; e.g. mirror_plane/delta_depth_precise/...] / [sample id].extension
- For Scannet and Matterport3D mirror data is stored as [data type; e.g. mirror_plane/delta_depth_precise/...] / [scene_id] / [sample id].extension

The sample's mirror 3D plane information is saved in a single JSON file. The data is saved as:

```shell

[
    { # one instance's 3D plane annotation
        "plane":[ # mirror plane parameter [a, b, c, d] for 3D mirror plane ax + by + cz + d = 0; ( y axis points upward; -z axis points to the front)
            -3.908371765225202e-05,
            -9.067424129382443e-06,
            -0.00020693446719167178,
            0.9999999777841854
        ],
        "normal":[ # mirror plane's normal (normalized to unit length)
            -0.1854170852709031,
            -0.04301677153499429,
            -0.9817179135863564
        ],
        "mask_id":1 # instance id in the 8-bit segmentation mask
    },
    {
        "plane":[ # the other instance's 3D plane annotation
            -3.724353384792313e-05,
            9.92165522022929e-07,
            -0.00020842300324026505,
            0.999999977585893
        ],
        "normal":[
            -0.175903777717696,
            0.004686066154670644,
            -0.9843962117809258
        ],
        "mask_id":2
    }
]
```


## Generate refined depth map

### STEP 1: download the source data 

To generate a refined depth map, please download the relevant source data and put it under the unzipped folder:

- For the Matterport3D dataset, please put the `matterport_render_depth`, `undistorted_color_images` and `undistorted_depth_images` folder under `mp3d` folder

- For the NYUv2-small dataset, please put the `color` and `depth` folder under `nyu` folder. We use `mirror3d/utils/export_mat_image.py` to generate color and depth frames from the .mat file NYUv2 provides.
  
- For the ScanNet dataset, please put the `scannet_extracted` and `scannet_frames_25k` folder under `scannet` folder

### STEP 2: generate symlinks for mirror samples' RGBD images

Please run the following command to create symlinks to the mirror samples' original color image, sensor depth map and mesh depth map:

```python
python mirror3d/dataset/gen_synlink.py --unzipped_folder_path [the path to the mp3d/ nyu/ scannet folder] 
```

### STEP 3 : generate refined depth map based on delta image

```python
python mirror3d/dataset/gen_refinedD_from_delta.py \
--unzipped_folder_path [the path to the mp3d/ nyu/ scannet folder] \
--mask_version [mirror mask version: precise (default) / coarse]
```

The generated refined depth map will be saved under the [zip_folder]. 



After STEP 1 ~ STEP 3, the data structure should be like:

- For **Matterport3D dataset**:

```shell
mp3d
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_depth_coarse
└── delta_depth_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── refined_meshD_coarse # refined mesh depth map (coarse version)
└── refined_meshD_precise # refined mesh depth map (coarse version)
└── mirror_plane
└── matterport_render_depth # source data
└── undistorted_color_images # source data
└── undistorted_depth_images # source data
└── raw_sensorD # mirror samples' sensor depth symlinks --- link to data under ./undistorted_depth_images
└── raw_meshD # mirror samples' mesh depth symlinks --- link to data under ./matterport_render_depth
└── mirror_color_images # mirror samples' color image symlinks --- link to data under ./undistorted_color_images

```


- For **NYUv2-small dataset**:

```shell
nyu
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_depth_coarse
└── delta_depth_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── mirror_plane
└── color # source data
└── depth # source data
└── raw_sensorD # mirror samples' sensor depth symlinks --- link to data under ./depth
└── mirror_color_images # mirror samples' color image symlinks --- link to data under ./color

```

- For **ScanNet dataset**:

```shell
scannet
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_depth_coarse
└── delta_depth_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── mirror_plane
└── scannet_extracted # source data
└── raw_sensorD # mirror samples' sensor depth symlinks --- link to data under ./scannet_extracted
└── mirror_color_images # mirror samples' color image symlinks --- link to data under ./scannet_extracted
```

### Optional : generate RGB instance mask

The mirror instances masks we provide in the zip files are 8-bit integer instance masks. If you want to generate RGB instance masks for visualization, you can run:

```shell
python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
--function 2 \
--input_txt [path to txt file] # Each line of this txt file should include information in format "[input integer mask path] [RGB mask output path]"

```

Here's a quick example of generating an RGB segmentation mask based on an 8-bit integer instance mask for an NYUv2 sample:
```shell
python mirror3d/annotation/plane_annotation/plane_annotation_tool.py \
--function 2 \
--input_txt docs/example/input_txt_example/get_color_mask.txt
```

## Data visualization
To check and visualize one sample's data, you can run:

```python
python mirror3d/visualization/check_sample_info.py \
--color_img_path [path to the sample's color image] \
--depth_img_path [path to the sample's depth image] \
--mask_img_path [path to the sample's integer mask] \
--json_path [path to the sample's JSON file] \
--f [relevant focal length: 1074 for Matterport3D, 519 for NYUv2-small, 574 for ScanNet]

```

For example, here's a screemshot of the 3D visulization after running:

```python
python mirror3d/visualization/check_sample_info.py \
--color_img_path annotation/plane_annotation/example/nyu/mirror_color_images/664.jpg \
--depth_img_path annotation/plane_annotation/example/nyu/refined_sensorD_precise/664.png \
--mask_img_path annotation/plane_annotation/example/nyu/mirror_instance_mask_precise/664.png \
--json_path annotation/plane_annotation/example/nyu/mirror_plane/664.json \
--f 519
```

to check the NYUV2 sample (smaple id : 664):

![data-check](figure/check-demo.png)

The dark blue points are mirror area reconstruction based on the refined sensor depth map. The light blue mesh plane is generated based on the mirror plane parameter stored in `annotation/plane_annotation/example/nyu/mirror_plane/664.json`. The purple arrow is the mirror normal generated by mirror normal information stored in `annotation/plane_annotation/example/nyu/mirror_plane/664.json`.
