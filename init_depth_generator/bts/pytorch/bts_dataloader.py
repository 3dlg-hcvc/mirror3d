# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random
import json

from distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            # self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            # self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        self.focal_lengths = []



        if mode != 'train':
            if args.coco_input:
                root_paths = args.coco_val_root.split(",")
                coco_focal_len = args.coco_focal_len.split(",")
                self.filepaths = []
                for dataset_index, one_json in enumerate(args.coco_val.split(",")):
                    one_json = one_json.strip()
                    input_images = read_json(one_json)["images"]
                    for one_info in input_images: 
                        if args.refined_depth:
                            if args.mesh_depth: # mesh refine
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["mesh_refined_path"])])
                            else:  # hole refine
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["hole_refined_path"])])
                        else:
                            if args.mesh_depth: # mesh raw
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["mesh_raw_path"])])
                            else:# mesh raw hole raw
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["hole_raw_path"])])
                        self.focal_lengths.append(int(coco_focal_len[dataset_index]))
            else:
                with open(args.filenames_file_eval, 'r') as f:
                    self.filepaths = f.readlines()
        else:
            if args.coco_input:
                root_paths = args.coco_train_root.split(",")
                coco_focal_len = args.coco_focal_len.split(",")
                self.filepaths = []
                for dataset_index, one_json in enumerate(args.coco_train.split(",")):
                    one_json = one_json.strip()
                    for one_info in read_json(one_json)["images"]:
                        if args.refined_depth:
                            if args.mesh_depth: # mesh refine
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["mesh_refined_path"])])
                            else:  # hole refine
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["hole_refined_path"])])
                        else:
                            if args.mesh_depth: # mesh raw
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["mesh_raw_path"])])
                            else:# mesh raw hole raw
                                self.filepaths.append([os.path.join(root_paths[dataset_index], one_info["img_path"]), os.path.join(root_paths[dataset_index], one_info["hole_raw_path"])])

                        self.focal_lengths.append(int(coco_focal_len[dataset_index]))
            else:
                with open(args.filenames_file, 'r') as f:
                    self.filepaths = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        
    
    def __getitem__(self, idx):
        sample_path = self.filepaths[idx]
        if self.args.coco_input:
            focal = self.focal_lengths[idx]
        else:
            focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if self.args.coco_input:
                image_path = sample_path[0]
                depth_path = sample_path[1]
            else:
                if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                    image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[3])
                    depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[4])
                else:
                    image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])
            if self.args.coco_input:
                image = Image.open(image_path).resize((self.args.input_width,self.args.input_height), Image.NEAREST)
                depth_gt = Image.open(depth_path).resize((self.args.input_width,self.args.input_height), Image.NEAREST)
            else:
                image = Image.open(image_path)
                depth_gt = Image.open(depth_path)


                
            
            
            # if self.args.do_kb_crop is True:
            #     height = image.height
            #     width = image.width
            #     top_margin = int(height - 352)
            #     left_margin = int((width - 1216) / 2)
            #     depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            #     image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # # To avoid blank boundaries due to pixel registration
            # if self.args.dataset == 'nyu' and not self.args.coco_input:
            #     depth_gt = depth_gt.crop((43, 45, 608, 472))
            #     image = image.crop((43, 45, 608, 472))
    
            # if self.args.do_random_rotate is True:
            #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
            #     image = self.rotate_image(image, random_angle)
            #     depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            depth_gt = depth_gt / self.args.depth_shift


            # image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,'image_path':sample_path[0],'gt_depth_path':sample_path[1],'gt_depth_path':sample_path[1]}
        
        else:
            if not self.args.coco_input:
                if self.mode == 'online_eval':
                    data_path = self.args.data_path_eval
                else:
                    data_path = self.args.data_path

            if self.args.coco_input:
                if sample_path[0].find("distorted") > 0:
                    image_path = sample_path[0]
                else:
                    image_path = sample_path[0]
            else:
                image_path = os.path.join(data_path, "./" + sample_path.split()[0])

            if self.args.coco_input:
                image = np.asarray(Image.open(image_path).resize((self.args.input_width,self.args.input_height)), dtype=np.float32) / 255.0 
            else:
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                if self.args.coco_input:
                    if sample_path[1].find("distorted") > 0:
                        depth_path = sample_path[1]
                    else:
                        depth_path = sample_path[1]
                else:
                    gt_path = self.args.gt_path_eval
                    depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                try:
                    if self.args.coco_input:
                        depth_gt = Image.open(depth_path).resize((self.args.input_width,self.args.input_height), Image.NEAREST)
                    else:
                        depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    depth_gt = depth_gt / self.args.depth_shift

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'image_path':sample_path[0],'gt_depth_path':sample_path[1]}
            else:
                sample = {'image': image, 'focal': focal,'image_path':sample_path[0], 'gt_depth_path':sample_path[1]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filepaths)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal, 'image_path':sample['image_path'],'gt_depth_path':sample['gt_depth_path']}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'image_path':sample['image_path'],'gt_depth_path':sample['gt_depth_path']}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'image_path':sample['image_path'],'gt_depth_path':sample['gt_depth_path']}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
