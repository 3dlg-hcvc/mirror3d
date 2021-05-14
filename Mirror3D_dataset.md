# Mirror3D 

Mirror3D is a large-scale 3D mirror plane annotation dataset based on three popular RGBD datasets ([Matterpot3D](https://niessner.github.io/Matterport/),[NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [ScanNet](http://www.scan-net.org/)) containing 4,852 mirror instance masks and 3D planes.

Please visit our [project website]() for updates and to browse the data.


## Mirror3D Data

If you would like to use our data, you must obtain access to the [Matterpot3D](https://niessner.github.io/Matterport/) and [ScanNet](http://www.scan-net.org/)) dataset.  Please send an email to the dataset organizer(s) to confirm your agreement and cc Jiaqi Tan (jiaqit[at]sfu.ca). You will get download links for the data from the dataset that you got approval.

## Mirror3D Data Organization

Mirror3D dataset includes mirror segmentation masks, mirror plane parameters, and refined mirror depth map over 4,852 RGBD images.


The downloaded mirror data are stored under the following structure:


###NYUv2-small 

For NYUv2-small, the unzipped package's structure is: 


```
nyu
└── with_mirror # mirror data
    ├── coarse
    |   ├── hole_raw_depth # depth maps captured by the depth sensor
    |   ├── hole_refined_depth # refined sensor depth maps
    |   ├── img_info # .json file with mirror plane's information; name by mirror color image
    |   ├── instance_mask # instance level mirror semantic mask
    |   └── raw # mirror color images
    └── precise
        ├── hole_raw_depth
        ├── hole_refined_depth
        ├── img_info
        ├── instance_mask
        └── raw
    
```
###Matterport3d
For Matterport3d, the unzipped package's structure is: 

```
m3d
├── only_mask # mirror data that only contains mirror masks' information; depth images for these samples are too noisy to annotate. 
│   ├── coarse
│   |   ├── instance_mask # instance level mirror semantic mask
│   |   └── raw # mirror color images
│   └── precise
│       ├── instance_mask
│       └── raw
└── with_mirror
    ├── coarse
    |   ├── hole_raw_depth # depth maps captured by the depth sensor
    |   ├── hole_refined_depth # refined sensor depth maps
    |   ├── mesh_raw_depth # mesh depth maps generated from the 3D mesh reconstruction
    |   ├── mesh_refined_depth # refined mirror mesh depth maps
    |   ├── img_info # .json file with mirror plane's information; name by mirror color image
    |   ├── instance_mask # instance level mirror semantic mask
    |   └── raw # mirror color images
    └── precise
        ├── hole_raw_depth
        ├── hole_refined_depth
        ├── mesh_raw_depth
        ├── mesh_refined_depth
        ├── img_info
        ├── instance_mask
        └── raw
```
###Scannet

For Scannet, the unzipped package's structure is: 

```
scannet
├── only_mask # mirror data that only contains mirror masks' information; depth images for these samples are too noisy to annotate. 
│   ├── coarse
│   |   ├── instance_mask
│   |   └── raw
│   └── precise
│       ├── instance_mask
│       └── raw
└── with_mirror
    ├── coarse
    |   ├── hole_raw_depth # depth maps captured by the depth sensor
    |   ├── hole_refined_depth # refined sensor depth maps
    |   ├── img_info # .json file with mirror plane's information; name by mirror color image
    |   ├── instance_mask # instance level mirror semantic mask
    |   └── raw # mirror color images
    └── precise
        ├── hole_raw_depth
        ├── hole_refined_depth
        ├── img_info
        ├── instance_mask
        └── raw
```

#### JSON files for Mirror Plane Information

Here, one mirror sample's plane parameters information is saved in a single JSON file. The information is saved in a format:

```json
{
    "0_0_128":{ # 0_0_128 is the instance id (R_G_B of the semantic mask)
        "plane_parameter":[ # mirror plane parameter in 3D; here y axis points upward, -z axis points to the front
            0.00025589483339543795,
            6.575998812963738e-05,
            -0.00026007778514475276,
            0.9999999312764996
        ],
        "mirror_normal":[ # mirror plane's normal 
            0.00025589483339543795,
            6.575998812963738e-05,
            -0.00026007778514475276
        ]
    }
}

```


## Training Data Structure

To train or test the models with the .json input file we provided, you need to download the original data from the relevant dataset and store these original data under the following structure:

###NYUv2-small

```
nyu
├── original_dataset # please create a folder named "original_dataset" and put the original dataset's data under this folder if you want to train our models.
│   ├── color
│   └── depth
└── with_mirror # downloaded mirror data
        
```

###Matterport3d
```
m3d
├── original_dataset # please create a folder named "original_dataset" and put the original dataset's data under this folder if you want to train our models.
│   ├── matterport_render_depth
│   ├── undistorted_color_images
│   └── undistorted_depth_images
├── only_mask # downloaded mirror data
└── with_mirror # downloaded mirror data
        
```

###Scannet

```
scannet
├── original_dataset # please create a folder named "original_dataset" and put the original dataset's data under this folder if you want to train our models.
│   └── scannet_frames_25k
├── only_mask # downloaded mirror data
└── with_mirror # downloaded mirror data
        
```
