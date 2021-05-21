# Mirror3D 

Mirror3D is a large-scale 3D mirror plane annotation dataset based on three popular RGBD datasets ([Matterpot3D](https://niessner.github.io/Matterport/),[NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [ScanNet](http://www.scan-net.org/)) containing 7,011 mirror instance masks and 3D planes.

Please visit our [project website]() for updates and to browse the data.



## Mirror3D Data Organization

The unzipped mirror data we provided are stored in the following structures:


### NYUv2-small (nyu.zip) / ScanNet (scannet.zip) / Matterport3d (m3d.zip)


```
nyu/scannet/m3d
├── mirror_instance_mask_coarse # stores coarse instance-level mirror segmentation mask
└── mirror_instance_mask_precise # stores precise instance-level mirror segmentation mask
└── delta_image_coarse # stores delta image to generate the coarse refined depth map
└── delta_image_precise # stores delta image to generate the precise refined depth map
└── mirror_plane # stores the mirror plane parameter information 
```

The sample's mirror 3D plane information is saved in a single JSON file. The data is saved as:

```
{
    "AF8080":{ # AF8080 is the instance id in hexadecimal on the semantic mask
        "plane_parameter":[ # mirror plane parameter in 3D; here y axis points upward, -z axis points to the front
            0.00025589483339543795,
            6.575998812963738e-05,
            -0.00026007778514475276,
            0.9999999312764996
        ],
        "mirror_normal":[ # mirror plane's normal (normalized to a unit vector)
            0.025589483339543795,
            6.575998812963738e-02,
            -0.026007778514475276
        ]
    }
}

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

Then you can get the generated refined depth map in the relevant folder.

The final data structure will be like:

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

```


- For **NYUv2-small dataset**:

```shell
m3d
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_image_coarse
└── delta_image_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── mirror_plane
└── color # source data
└── depth # source data

```

- For **ScanNet dataset**:

```shell
m3d
├── mirror_instance_mask_coarse
└── mirror_instance_mask_precise
└── delta_image_coarse
└── delta_image_precise
└── refined_sensorD_coarse # refined sensor depth map (coarse version)
└── refined_sensorD_precise # refined sensor depth map (precise version)
└── mirror_plane
└── scannet_frames_25k # source data

```


To validate the correctness of the generated depth, you can run:

```python
python visualization/check_sample_info.py --data_root_path [path to the unzipped m3d/nyu/scannet folder] --json_path [any JSON file stored under the mirror_plane foler] --f [relevant focal length: 1074 for Matterport3D, 519 for NYUv2-small, 574 for ScanNet]

```

Demo: after running `python visualization/check_sample_info.py --data_root_path ./nyu --json_path ./nyu/mirror_plane/1003.json --f 519` , you can visulize the NYUV2 sample 1003 like:

![data-check](figure/check-demo.png)

Here, the black line is the mirror normal (perpendicular to the mirror plane), the light blue mesh is the mirror plane.
