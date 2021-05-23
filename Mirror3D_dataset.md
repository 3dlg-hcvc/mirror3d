# Mirror3D 

Mirror3D is a large-scale 3D mirror plane annotation dataset based on three popular RGBD datasets ([Matterpot3D](https://niessner.github.io/Matterport/),[NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [ScanNet](http://www.scan-net.org/)) containing 7,011 mirror instance masks and 3D planes.

Please visit our [project website]() for updates and to browse the data.



## Mirror3D Data Organization

The unzipped mirror data we provided are stored in the following structures:


### NYUv2-small (nyu.zip) / ScanNet (scannet.zip) / Matterport3d (m3d.zip)


```shell
nyu/scannet/m3d
├── mirror_instance_mask_coarse # stores coarse instance-level mirror segmentation mask
└── mirror_instance_mask_precise # stores precise instance-level mirror segmentation mask
└── delta_image_coarse # stores delta image to generate the coarse refined depth map
└── delta_image_precise # stores delta image to generate the precise refined depth map
└── mirror_plane # stores the mirror plane parameter information 
```

The sample's mirror 3D plane information is saved in a single JSON file. The data is saved as:

```shell

[
    { # one instance's 3D plane annotation
        "plane":[ # mirror plane parameter in 3D; here y axis points upward, -z axis points to the front
            -8.45127360669912e-05,
            -3.9755436110599056e-07,
            -0.00010689417965837903,
            0.9999999907155368
        ],
        "normal":[ # mirror plane's normal (normalized to unit length)
            -0.6201957221843089,
            -0.0029174480151513255,
            -0.784441683416532
        ],
        "mask_id":"008000" # 008000 is the instance id in hexadecimal on the semantic mask
    },
    { # the other instance's 3D plane annotation
        "plane":[
            0.0001230891443982547,
            1.7867283127929448e-06,
            -0.0001010830611631414,
            0.9999999873140424
        ],
        "normal":[
            0.7727573205745162,
            0.011217133650074177,
            -0.6346024735306119
        ],
        "mask_id":"000080"
    }
]

```

## Generate refined depth map

To generate a refined depth map, please download the relevant source data and put it under the unzipped folder:

- For Matterport3D dataset, please put the `matterport_render_depth`, `undistorted_color_images` and `undistorted_depth_images` folder under `m3d` folder

- For NYUv2-small dataset, please put the `color` and `depth` folder under `nyu` folder
  
- For ScanNet dataset, please put the `scannet_frames_25k` folder under `scannet` folder

Then run:

```python
python ***.py --zip_folder [the path to the m3d/ nyu/ scannet folder] 
```

The generated refined depth map will be saved under the [zip_folder]. 

To generate our networks training input, please run ***.py to create symlinks to the mirror samples' original color image, sensor depth map and mesh depth map:

```
python ***.py --dataset_main_folder [the path to the m3d/ nyu/ scannet folder] 
```

Finally, the data structure should be like:

- For **Matterport3D dataset**:

```shell
m3d
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_image_coarse
└── delta_image_precise
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
└── delta_image_coarse
└── delta_image_precise
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
└── delta_image_coarse
└── delta_image_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── mirror_plane
└── scannet_frames_25k # source data
└── raw_sensorD # mirror samples' sensor depth symlinks --- link to data under ./scannet_frames_25k
└── mirror_color_images # mirror samples' color image symlinks --- link to data under ./scannet_frames_25k
```


To validate the correctness of the generated depth map, you can run:

```python
python visualization/check_sample_info.py --data_root_path [path to the unzipped m3d/nyu/scannet folder] --json_path [any JSON file stored under the mirror_plane foler] --f [relevant focal length: 1074 for Matterport3D, 519 for NYUv2-small, 574 for ScanNet]

```

For example, here's a screemshot of the 3D visulization after running `python visualization/check_sample_info.py --data_root_path ./nyu --json_path ./nyu/mirror_plane/1003.json --f 519` , for the NYUV2 sample (smaple id : 1003):

![data-check](figure/check-demo.png)

The black line is the mirror normal (perpendicular to the mirror plane), the light blue mesh is the mirror plane.
