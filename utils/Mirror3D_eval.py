import torch
import cv2
import os
import numpy as np
import math
import argparse
import sys
from utils.general_utlis import *
import logging

class Mirror3d_eval():
    def __init__(self, train_with_refD, logger=None, Input_tag="Input_tag", method_tag="method_tag",width=640, height=480):
        self.m_nm_all = torch.zeros(27)
        self.cnt = 0
        self.train_with_refD = train_with_refD
        self.logger = logger
        self.Input_tag = Input_tag
        self.method_tag = method_tag
        self.width = width
        self.height = height

        if self.train_with_refD == True: 
            self.Train_tag = "ref"
        elif self.train_with_refD == False:
            self.Train_tag = "raw"
        else:
            self.Train_tag = "*"
    
    def reset_setting(self,train_with_refD, logger=None, Input_tag="Input_tag", method_tag="method_tag",width=640, height=480):
        self.m_nm_all = torch.zeros(27)
        self.cnt = 0
        self.train_with_refD = train_with_refD
        self.logger = logger
        self.Input_tag = Input_tag
        self.method_tag = method_tag
        self.width = width
        self.height = height

        if self.train_with_refD == True: 
            self.Train_tag = "ref"
        elif self.train_with_refD == False:
            self.Train_tag = "raw"
        else:
            self.Train_tag = "*"


    def print_mirror3D_score(self):

        eval_measures_cpu = self.m_nm_all
        eval_measures_cpu /= self.cnt
        print('Computing errors for {} eval samples'.format(int(self.cnt)))

        # print title
        print("{:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12} \\\\".format(
        'Input', "Train", "Method", "Region", "RMSE", "s-RMSE", "Rel", "SSIM", '$d_{1.05}$', '$d_{1.12}$', '$d_{1.25}$', '$d_{1.25^2','$d_{1.25^3}$', "Count"))
        if self.logger:
            self.logger.info("{:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12} \\\\".format(
        'Input', "Train", "Method", "Region", "RMSE", "s-RMSE", "Rel", "SSIM", '$d_{1.05}$', '$d_{1.12}$', '$d_{1.25}$', '$d_{1.25^2}$','$d_{1.25^3}$', "Count"))

        # print mirror area score
        print_line = "{:>12}& {:>12}& {:>12}& {:>12}& ".format(self.Input_tag, self.Train_tag, self.method_tag, "mirror")
        for i in range(0,9):
            print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
        print_line += '{:>12} \\\\'.format(int(self.cnt))
        print(print_line)
        if self.logger:
            self.logger.info(print_line)

        # print non-mirror area score
        print_line = "{:>12}& {:>12}& {:>12}& {:>12}& ".format(self.Input_tag, self.Train_tag, self.method_tag, "non-mirror")
        for i in range(9,18):
            print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
        print_line += '{:>12} \\\\'.format(int(self.cnt))
        print(print_line)
        if self.logger:
            self.logger.info(print_line)

        # print all area score
        print_line = "{:>12}& {:>12}& {:>12}& {:>12}& ".format(self.Input_tag, self.Train_tag, self.method_tag, "all")
        for i in range(18,27):
            print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
        print_line += '{:>12} \\\\'.format(int(self.cnt))
        print(print_line)
        if self.logger:
            self.logger.info(print_line)

    def compute_and_update_rmse_srmse(self, pred_depth, depth_shift, color_image_path):

        if color_image_path.find("m3d") > 0 and "mesh" not in self.Train_tag:
            self.Train_tag = self.Train_tag.replace("ref","mesh-ref")
            self.Train_tag = self.Train_tag.replace("raw","mesh")

        def compute_errors(gt, pred, mask): #! gt and pred are in m

            min_depth_eval = 1e-3
            max_depth_eval = 10

            pred[pred < min_depth_eval] = min_depth_eval
            # pred[pred > max_depth_eval] = max_depth_eval
            pred[np.isinf(pred)] = max_depth_eval

            gt[np.isinf(gt)] = 0
            gt[np.isnan(gt)] = 0

            valid_mask = gt >  min_depth_eval #  np.logical_and(gt > min_depth_eval)#, gt < max_depth_eval
            scale = np.sum(pred[valid_mask]*gt[valid_mask])/np.sum(pred[valid_mask]**2)
            valid_mask = np.logical_and(valid_mask, mask)

            gt = gt[valid_mask]
            pred = pred[valid_mask]

            if valid_mask.sum() == 0 :
                return False

            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())


            scaled_rms = np.sqrt(((scale * pred-gt)**2).mean())

            return rmse, scaled_rms, 0, 0, 0, 0, 0, 0, 0


        mask_path = rreplace(color_image_path, "raw","instance_mask")
        if not os.path.exists(mask_path):
            return 

        if color_image_path.find("m3d") > 0:
            if os.path.exists(rreplace(color_image_path.replace("raw", "mesh_refined_depth"),"i","d")):
                refD_gt_depth_path = rreplace(color_image_path.replace("raw", "mesh_refined_depth"),"i","d")
            elif os.path.exists(rreplace(color_image_path.replace("raw", "hole_refined_depth"),"i","d")):
                refD_gt_depth_path = rreplace(color_image_path.replace("raw", "hole_refined_depth"),"i","d")
            else:
                return
        else:
            if os.path.exists(color_image_path.replace("raw", "mesh_refined_depth")):
                refD_gt_depth_path = color_image_path.replace("raw", "mesh_refined_depth")
            elif os.path.exists(color_image_path.replace("raw", "hole_refined_depth")):
                refD_gt_depth_path = color_image_path.replace("raw", "hole_refined_depth")
            else:
                return
        
        depth_shift = np.array(depth_shift)
        refD_gt_depth = cv2.resize(cv2.imread(refD_gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
        refD_gt_depth = np.array(refD_gt_depth) / depth_shift
        pred_depth = np.array(pred_depth)

        mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        refD_gt_depth = cv2.resize(refD_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        
        mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask>0)
        non_mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask==False)
        all_image_error = compute_errors(refD_gt_depth, pred_depth, True)


        print("{:.3f} {:.3f} {:.3f} {}".format(mirror_error[0], non_mirror_error[0], all_image_error[0],refD_gt_depth_path)) # TODO debug delete later
        if all_image_error == False or mirror_error == False or non_mirror_error == False:
            return 
        
        one_m_nm_all = mirror_error + non_mirror_error + all_image_error

        self.m_nm_all += torch.tensor(one_m_nm_all)
        self.cnt += 1

        return 


    def compute_and_update_mirror3D_metrics(self, pred_depth, depth_shift, color_image_path):

        if color_image_path.find("m3d") > 0 and "mesh" not in self.Train_tag:
            self.Train_tag = self.Train_tag.replace("ref","mesh-ref")
            self.Train_tag = self.Train_tag.replace("raw","mesh")

        def compute_errors(gt, pred, mask): #! gt and pred are in m

            min_depth_eval = 1e-3
            max_depth_eval = 10

            pred[pred < min_depth_eval] = min_depth_eval
            pred[np.isinf(pred)] = max_depth_eval

            gt[np.isinf(gt)] = 0
            gt[np.isnan(gt)] = 0

            valid_mask = gt >  min_depth_eval #  np.logical_and(gt > min_depth_eval)#, gt < max_depth_eval
            scale = np.sum(pred[valid_mask]*gt[valid_mask])/np.sum(pred[valid_mask]**2)
            valid_mask = np.logical_and(valid_mask, mask)

            SSIM_obj = SSIM()
            ssim_map = SSIM_obj.forward(torch.tensor(pred*valid_mask.astype(int)).unsqueeze(0).unsqueeze(0), torch.tensor(gt*valid_mask.astype(int)).unsqueeze(0).unsqueeze(0))
            ssim = ssim_map[valid_mask].mean()

            gt = gt[valid_mask]
            pred = pred[valid_mask]

            if valid_mask.sum() == 0 :
                return False

            thresh = np.maximum((gt / pred), (pred / gt))
            d125 = (thresh < 1.25).mean()
            d125_2 = (thresh < 1.25 ** 2).mean()
            d125_3 = (thresh < 1.25 ** 3).mean()
            d105 = (thresh < 1.05).mean()
            d110 = (thresh < 1.10).mean()

            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())

            rel = np.mean((abs(gt - pred)) / gt)

            err = np.log(pred) - np.log(gt)
            silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

            err = np.abs(np.log10(pred) - np.log10(gt))
            log10 = np.mean(err)

            scaled_rms = np.sqrt(((scale * pred-gt)**2).mean())

            return rmse, scaled_rms, rel, ssim.item(), d105, d110, d125, d125_2, d125_3


        mask_path = rreplace(color_image_path, "raw","instance_mask")
        if not os.path.exists(mask_path):
            return 

        if color_image_path.find("m3d") > 0:
            if os.path.exists(rreplace(color_image_path.replace("raw", "mesh_refined_depth"),"i","d")):
                refD_gt_depth_path = rreplace(color_image_path.replace("raw", "mesh_refined_depth"),"i","d")
            elif os.path.exists(rreplace(color_image_path.replace("raw", "hole_refined_depth"),"i","d")):
                refD_gt_depth_path = rreplace(color_image_path.replace("raw", "hole_refined_depth"),"i","d")
            else:
                return
        else:
            if os.path.exists(color_image_path.replace("raw", "mesh_refined_depth")):
                refD_gt_depth_path = color_image_path.replace("raw", "mesh_refined_depth")
            elif os.path.exists(color_image_path.replace("raw", "hole_refined_depth")):
                refD_gt_depth_path = color_image_path.replace("raw", "hole_refined_depth")
            else:
                return
        
        depth_shift = np.array(depth_shift)
        refD_gt_depth = cv2.resize(cv2.imread(refD_gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
        refD_gt_depth = np.array(refD_gt_depth) / depth_shift
        pred_depth = np.array(pred_depth)

        mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        refD_gt_depth = cv2.resize(refD_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        
        mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask>0)
        non_mirror_error = compute_errors(refD_gt_depth, pred_depth, mirror_mask==False)
        all_image_error = compute_errors(refD_gt_depth, pred_depth, True)

        if all_image_error == False or mirror_error == False or non_mirror_error == False:
            return 
        
        one_m_nm_all = mirror_error + non_mirror_error + all_image_error

        self.m_nm_all += torch.tensor(one_m_nm_all)
        self.cnt += 1

        return 

    def save_result(self, main_output_folder, pred_depth, depth_shift, color_img_path):


        if os.path.exists(color_img_path.replace("raw", "mesh_refined_depth")):
            refD_gt_depth_path = color_img_path.replace("raw", "mesh_refined_depth")
        elif os.path.exists(color_img_path.replace("raw", "hole_refined_depth")):
            refD_gt_depth_path = color_img_path.replace("raw", "hole_refined_depth")
        elif os.path.exists(rreplace(color_img_path.replace("raw", "mesh_refined_depth"),"i","d")):
            refD_gt_depth_path = rreplace(color_img_path.replace("raw", "mesh_refined_depth"),"i","d")
        elif os.path.exists(rreplace(color_img_path.replace("raw", "hole_refined_depth"),"i","d")):
            refD_gt_depth_path = rreplace(color_img_path.replace("raw", "hole_refined_depth"),"i","d")
        else:
            return

        pred_depth = np.array(pred_depth)
        depth_shift = np.array(depth_shift)

        info_txt_save_path = os.path.join(main_output_folder, "color_mask_gtD_predD.txt")
        pred_depth_scaled = pred_depth * depth_shift
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
                    
        predD_save_folder = os.path.join(main_output_folder,"pred_depth")
        mask_path = color_img_path.replace("raw", "instance_mask")

        if not os.path.exists(mask_path):
            mask_path = None
        os.makedirs(predD_save_folder, exist_ok=True)
        if "scannet" not in color_img_path:
            depth_np_save_path = os.path.join(predD_save_folder, refD_gt_depth_path.split("/")[-1])
        else:
            depth_np_save_path = os.path.join(predD_save_folder, "{}_{}".format(refD_gt_depth_path.split("/")[-2], refD_gt_depth_path.split("/")[-1]))
        
        cv2.imwrite(depth_np_save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        with open(info_txt_save_path, "a") as file:
            file.write("{} {} {} {}".format(color_img_path, mask_path, refD_gt_depth_path, depth_np_save_path))
            file.write("\n")


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, eps=1e-5):
        super(SSIM, self).__init__()
        self.eps = eps
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.squeeze()

    def forward(self, pred, gt):

        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img2[img2 < self.eps] = 0
        img1[img2 < self.eps] = 0

        (_, channel, _, _) = img1.size()


        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)



class Mirror_seg_eval():
    def __init__(self, width=640, height=480):
        self.f_measure_list = []
        self.IOU_list = []
        self.MAE_list = []
        self.width = width
        self.height = height

    def reset_setting(self,width=640, height=480):
        self.f_measure_list = []
        self.IOU_list = []
        self.MAE_list = []
        self.width = width
        self.height = height

    def print_seg_score(self):

        print("|{:<25} : {:<5}|".format("evalution smaple num", len(self.f_measure_list)))
        print("|{:<25} : {:<5.3f}|".format("F measure", np.mean(self.f_measure_list)))
        print("|{:<25} : {:<5.3f}|".format("MAE", np.mean(self.MAE_list)))
        print("|{:<25} : {:<5.3f}|".format("IOU", np.mean(self.IOU_list)))


    def compute_and_update_seg_metrics(self, pred_mask, gt_mask):
        pred_mask = cv2.resize(pred_mask.astype("uint8"), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask.astype("uint8"), (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
        self.IOU_list.append(get_IOU(pred_mask, gt_mask))
        self.MAE_list.append(get_MAE(pred_mask, gt_mask))
        self.f_measure_list.append(get_f_measure(pred_mask, gt_mask))


    def get_results(self):
        return self.IOU_list, self.f_measure_list, self.MAE_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument( 
        '--info_list', default="/project/3dlg-hcvc/jiaqit/cvpr/R1_rerun/to_eval.txt", type=str, help="infomation txt contrains color_mask_gtD_predD.txt") 
    parser.add_argument( 
        '--line', default=0, type=int, help="line to processs from info_list") 
    parser.add_argument( 
        '--depth_shift', default=4000, type=int, help="depth shirt") 
    args = parser.parse_args()

    
    info_list_path = read_txt(args.info_list)[args.line]
    
    if "refinedD" in info_list_path:
        train_with_refD = True
    elif "rawD" in info_list_path:
        train_with_refD = False
    else:
        train_with_refD = None
    log_file_save_path = info_list_path.replace("color_mask_gtD_predD.txt","eval_output.log")
    logging.basicConfig(filename=log_file_save_path, filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)


    if info_list_path.find("saic") > 0:
        Input_tag = "RGBD"
        method_tag = "saic+M3DNet"
    elif info_list_path.find("bts") > 0:
        Input_tag = "RGB"
        method_tag = "bts+M3DNet"
    elif info_list_path.find("vnl") > 0:
        Input_tag = "RGB"
        method_tag = "vnl+M3DNet"
    elif info_list_path.find("planercnn"):
        Input_tag = "RGB"
        method_tag = "planercnn"
    else:
        Input_tag = "RGB"
        method_tag = "Mirror3DNet"

    eval_fun = Mirror3d_eval(train_with_refD, logger=logging, Input_tag=Input_tag, method_tag=method_tag,width=640, height=480)

    for one_color_mask_gtD_predD in read_txt(info_list_path):
        color_path, mask_path, gtD_path, predD_path = one_color_mask_gtD_predD.split()
        predD = cv2.imread(predD_path, cv2.IMREAD_ANYDEPTH) / args.depth_shift
        eval_fun.compute_and_update_rmse_srmse(predD, args.depth_shift, color_path)
    print("input info path : {}".format(info_list_path))
    eval_fun.print_mirror3D_score()
