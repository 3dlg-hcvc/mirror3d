## We test three methods on our dataset:

- [BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cogaplex-bts/bts)
- [VNL: Enforcing geometric constraints of virtual normal for depth prediction](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)
- [saic : Decoder Modulation for Indoor Depth Completion](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37)

## To test these three methods on our dataset:


- STEP 1: Please clone the code from their original repository
	- [BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cogaplex-bts/bts)
	- [VNL: Enforcing geometric constraints of virtual normal for depth prediction](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)
	- [saic : Decoder Modulation for Indoor Depth Completion](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37)

- STEP 2: Merge our modified scripts with the relevant original repository by copying files provided in `Mirror3D/init_depth_generator1` to the appropriate repository. (We mainly changed the data_loader, training, and testing script.) 
	- For example, if you want to test BTS, please clone the [BTS original repository](https://github.com/cogaplex-bts/bts). And copy files under `Mirror3D/init_depth_generator/bts` to the cloned BTS's workstation. 

- STEP 3: Remeneter to `export PYTHONPATH=[Mirror3D repository absolute path]`, during training and testing, we need to use the function provided in `Mirror3D/utils`

- STEP 4: To train or test BTS/ VNL/ saic on our dataset and reproduce the result reported in our paper, please run the relevant command provide in (TODO)
