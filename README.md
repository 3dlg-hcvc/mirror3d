# [Mirror3D: Depth Refinement for Mirror Surfaces](https://3dlg-hcvc.github.io/mirror3d)
                            




## Preparation for all implementations

```shell
mkdir workspace && cd workspace

### Put data under dataset folder
mkdir dataset

### Clone this repo and pull all submodules
git clone --recursive https://github.com/3dlg-hcvc/mirror3d.git

```

## Environment Setup

- python 3.7.4

```shell
### Install packages 
cd mirror3d && pip install -e .

### Setup Detectron2
python -m pip install git+https://github.com/facebookresearch/detectron2.git
```

## Dataset

Please refer to [Mirror3D Dataset](docs/Mirror3D_dataset.md) for instructions on how to prepare mirror data. Please visit our [project website](https://3dlg-hcvc.github.io/mirror3d) for updates and to browse more data.


<table width="80%" border="0" >


<tr>
<th>
Matterport3D
</th>
<th>
ScanNet
</th>
<th>
NYUv2
</th>
</tr>

<tr>
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/mp3d-data.png" />
</td>
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/scannet-data.png" />
</td>
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/nyu-data.png" />
</td>
</tr>


<tr color="white">
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/mp3d-data.gif" />
</td>
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/scannet-data.gif" />
</td>
<td align="center" valign="center" style="width:30%;height: 250px;">
<img width=auto height="200" src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/nyu-data.gif" />
</td>
</tr>



</table>



### Mirror annotation tool
Please refer to [User Instruction](docs/user_instruction.md) for instructions on how to annotate mirror data. 


## Models

### Mirror3DNet PyTorch Implementation

Mirror3DNet architecture can be used for either an RGB image or an RGBD image input. For an RGB input, we refine the depth of the predicted depth map D<sub>pred</sub> output by a depth estimation module. For RGBD input, we refine a noisy input depth D<sub>noisy</sub>.

<p align="center">
    <img src="http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/img/readme_img/network-arch-cr-new.jpg">
</p>

Please check [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet) for our network's pytorch implementation. 

### Initial Depth Generator Implementation

We test three methods on our dataset:

- [BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cogaplex-bts/bts)
- [VNL: Enforcing geometric constraints of virtual normal for depth prediction](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)
- [saic : Decoder Modulation for Indoor Depth Completion](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37)

We updated the dataloader and the main train/test script in the original repository to support our input format. 

## Network input

Our network inputs are JSON files stored based on [coco annotation format](https://cocodataset.org/#home). Please download [network input json](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/network_input_json.zip) to train and test our models. 

## Training

To train our models please run:

```shell
cd workspace

### Download network input json
wget http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/network_input_json.zip
unzip network_input_json.zip

### Get R-50.pkl from detectron2 to train Mirror3DNet and PlaneRCNN
mkdir checkpoint && cd checkpoint
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

cd ../mirror3d

### Train on NYUv2 mirror data
bash script/nyu_train.sh

### Train on Matterport3D mirror data
bash script/mp3d_train.sh
```

By default, we put the unzipped data and network input packages under `../dataset`. Please change the relevant configuration if you store the data in different directories. Output checkpoints and tensorboard log files are saved under `--log_directory`.

## Inference

```shell
### Download all model zoo
cd workspace
wget http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint.zip
unzip checkpoint.zip

### Download network input json
wget http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/network_input_json.zip
unzip network_input_json.zip
cd mirror3d

### Inferece on NYUv2 mirror data
bash script/nyu_infer.sh

### Inferece on Matterport3D mirror data
bash script/mp3d_infer.sh
```

Output depth maps are saved under a folder named `pred_depth`. Optional: If you want to view all inference results on an html webpage, please run all steps in [mirror3d/visualization/result_visualization.py](https://github.com/3dlg-hcvc/mirror3d/blob/main/mirror3d/visualization/result_visualization.py).  


## Model Zoo

| **Source Dataset** | **Input** | **Train**            | Method                                                                                              | **Model Download**                                                                                                            |
|--------------------|-----------|----------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| NYUv2              | RGBD      | raw sensor depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_rawD.zip)                  |
| NYUv2              | RGBD      | refined sensor depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_refD.zip)                  |
| NYUv2              | RGB       | raw sensor depth     | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_nyu_v2_pytorch_densenet161.zip](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip) |
| NYUv2              | RGB       | refined sensor depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/bts_refD.zip)                    |
| NYUv2              | RGB       | raw sensor depth     | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [nyu_rawdata.pth](https://cloudstor.aarnet.edu.au/plus/s/7kdsKYchLdTi53p)                                                     |
| NYUv2              | RGB       | refined sensor depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/vnl_refD.zip)                    |
| Matterport3D       | RGBD      | raw sensor depth     | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_rawD.zip)   |
| Matterport3D       | RGBD      | refined sensor depth | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_refD.zip)   |
| Matterport3D       | RGBD      | raw sensor depth     | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_rawD.zip)       |
| Matterport3D       | RGBD      | refined sensor depth | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_refD.zip)       |
| Matterport3D       | RGBD      | raw sensor depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_rawD.zip)                 |
| Matterport3D       | RGBD      | refined sensor depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_refD.zip)                 |
| Matterport3D       | RGB       | *                    | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_normal_10.zip)   |
| Matterport3D       | RGB       | raw sensor depth     | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_rawD.zip)                   |
| Matterport3D       | RGB       | refined sensor depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_refD.zip)                   |
| Matterport3D       | RGB       | raw sensor depth     | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_rawD.zip)                   |
| Matterport3D       | RGB       | refined sensor depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_refD.zip)                   |
