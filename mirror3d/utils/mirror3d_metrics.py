import torch
import cv2
import os
import numpy as np
import math
from mirror3d.utils.general_utils import *
import time


class Mirror3dEval:
    def __init__(self, train_with_ref_d, logger=None, input_tag="input_tag", method_tag="method_tag", width=640,
                 height=480, dataset="nyu", dataset_root=""):
        self.m_nm_all_refD = torch.zeros(27)
        self.m_nm_all_rawD = torch.zeros(27)
        self.raw_cnt = 0
        self.ref_cnt = 0
        self.dataset_root = dataset_root
        self.train_with_refD = train_with_ref_d
        self.logger = logger
        self.input_tag = input_tag
        self.method_tag = method_tag
        self.width = width
        self.height = height
        self.dataset = dataset
        if self.train_with_refD:
            if self.dataset != "mp3d":
                self.Train_tag = "ref"
            else:
                self.Train_tag = "mesh-ref"
        elif not self.train_with_refD:
            if self.dataset != "mp3d":
                self.Train_tag = "raw"
            else:
                self.Train_tag = "mesh"
        else:
            self.Train_tag = "*"
        self.main_output_folder = "output/{}_{}_{}_{}_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                                                                 self.input_tag, self.Train_tag, self.method_tag,
                                                                 self.dataset)
        self.main_output_folder = self.main_output_folder.replace("*", "a")
        os.makedirs(self.main_output_folder, exist_ok=True)
        self.method_logFile_json_save_folder = "output"
        self.cal_std = True
        self.sample_name = []
        self.sample_score = dict()
        self.min_threshold_filter = True
        self.save_score_per_sample = True
        self.get_full_set = False
        self.to_create_folder = True

    def set_min_threshold_filter(self, min_threshold_filter):
        self.min_threshold_filter = min_threshold_filter

    def set_cal_std(self, cal_std):
        self.cal_std = cal_std

    def set_save_score_per_sample(self, save_score_per_sample):
        self.save_score_per_sample = save_score_per_sample

    def set_method_logFile_json_save_folder(self, folder):
        self.method_logFile_json_save_folder = folder

    def reset_setting(self, train_with_ref_d, logger=None, input_tag="input_tag", method_tag="method_tag", width=640,
                      height=480):
        self.m_nm_all_refD = torch.zeros(27)
        self.m_nm_all_rawD = torch.zeros(27)
        self.raw_cnt = 0
        self.ref_cnt = 0
        self.train_with_refD = train_with_ref_d
        self.logger = logger
        self.input_tag = input_tag
        self.method_tag = method_tag
        self.width = width
        self.height = height

        if self.train_with_refD:
            if self.dataset != "mp3d":
                self.Train_tag = "ref"
            else:
                self.Train_tag = "mesh-ref"
        elif not self.train_with_refD:
            if self.dataset != "mp3d":
                self.Train_tag = "raw"
            else:
                self.Train_tag = "mesh"
        else:
            self.Train_tag = "*"

    def save_as_table_format(self, eval_measures_cpu, compare_with_raw=False, compute_std=False):
        latex_method_tag = self.method_tag
        latex_method_tag.replace("BTS", "BTS~\cite{lee2019big}")
        latex_method_tag.replace("VNL", "VNL~\cite{yin2019enforcing}")
        latex_method_tag.replace("saic", "saic~\cite{senushkin2020decoder}")
        latex_method_tag.replace("PlaneRCNN", "PlaneRCNN~\\cite{liu2019planercnn}")

        tag = "{},{},{}".format(self.input_tag, self.Train_tag, latex_method_tag)
        table_one_line_result = dict()
        table_one_line_result[
            "RMSE,SSIM"] = "{:>5} & {:>10} & {:>45} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\".format(
            self.input_tag, self.Train_tag, latex_method_tag, eval_measures_cpu[0], eval_measures_cpu[9],
            eval_measures_cpu[18],
            eval_measures_cpu[3], eval_measures_cpu[12], eval_measures_cpu[21]
        )
        table_one_line_result[
            "RMSE,s-RMSE,Rel,SSIM"] = "{:>5} & {:>10} & {:>45} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} " \
                                      "& {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\".format(
            self.input_tag, self.Train_tag, latex_method_tag, eval_measures_cpu[0], eval_measures_cpu[9],
            eval_measures_cpu[18],
            eval_measures_cpu[1], eval_measures_cpu[10], eval_measures_cpu[19],
            eval_measures_cpu[2], eval_measures_cpu[11], eval_measures_cpu[20],
            eval_measures_cpu[3], eval_measures_cpu[12], eval_measures_cpu[21]
        )
        table_one_line_result["$d_{1.05}$,$d_{1.10}$,$d_{1.25}$,$d_{1.25^2}$,$d_{1.25^3}$"] = \
            "{:>5} & {:>10} & {:>45} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} " \
            "& {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\".format(
                self.input_tag, self.Train_tag, latex_method_tag,
                eval_measures_cpu[4], eval_measures_cpu[13], eval_measures_cpu[22],
                eval_measures_cpu[5], eval_measures_cpu[14], eval_measures_cpu[23],
                eval_measures_cpu[6], eval_measures_cpu[15], eval_measures_cpu[24],
                eval_measures_cpu[7], eval_measures_cpu[16], eval_measures_cpu[25],
                eval_measures_cpu[8], eval_measures_cpu[17], eval_measures_cpu[26]
            )
        table_one_line_result["main_output_folder"] = "{}_{}_{} {}".format(self.input_tag, self.Train_tag,
                                                                           latex_method_tag,
                                                                           os.path.abspath(self.main_output_folder))

        if compare_with_raw:
            one_name = "raw_result_minFilter_{}_full_{}.json".format(self.min_threshold_filter, self.get_full_set)
            save_name = "raw_{}_minFilter_{}_full_{}_result.json".format(self.dataset, self.min_threshold_filter,
                                                                         self.get_full_set)
        else:
            one_name = "ref_result_minFilter_{}_full_{}.json".format(self.min_threshold_filter, self.get_full_set)
            save_name = "ref_{}_minFilter_{}_full_{}_result.json".format(self.dataset, self.min_threshold_filter,
                                                                         self.get_full_set)

        if compute_std:
            save_name = "std_" + save_name

        os.makedirs(self.method_logFile_json_save_folder, exist_ok=True)
        method_log_file_json_save_path = os.path.join(self.method_logFile_json_save_folder, save_name)
        if os.path.exists(method_log_file_json_save_path):
            log_file_json = read_json(method_log_file_json_save_path)
        else:
            log_file_json = dict()
        if tag in log_file_json:
            if table_one_line_result not in log_file_json[tag]:
                log_file_json[tag].append(table_one_line_result)
        else:
            log_file_json[tag] = [table_one_line_result]
        save_json(method_log_file_json_save_path, log_file_json)
        print("update info file : {}".format(method_log_file_json_save_path))

        latex_temp_save_path = os.path.join(self.main_output_folder, one_name)
        save_json(latex_temp_save_path, table_one_line_result)
        print("latex result saved to : {}".format(latex_temp_save_path))

    def print_mirror3D_score(self):

        def print_all_scores(eval_measures_cpu, cnt):

            print('Computing errors for {} eval samples'.format(int(cnt)))

            # print title
            print(
                "{:>12}& {:>12}& {:>18}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12} \\\\".format(
                    'Input', "Train", "Method", "Region", "RMSE", "s-RMSE", "Rel", "SSIM", '$d_{1.05}$', '$d_{1.10}$',
                    '$d_{1.25}$', '$d_{1.25^2}$', '$d_{1.25^3}$', "Count"))
            if self.logger:
                self.logger.info(
                    "{:>12}& {:>12}& {:>18}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12}& {:>12} \\\\".format(
                        'Input', "Train", "Method", "Region", "RMSE", "s-RMSE", "Rel", "SSIM", '$d_{1.05}$',
                        '$d_{1.10}$', '$d_{1.25}$', '$d_{1.25^2}$', '$d_{1.25^3}$', "Count"))

            # print mirror area score
            print_line = "{:>12}& {:>12}& {:>18}& {:>12}& ".format(self.input_tag, self.Train_tag, self.method_tag,
                                                                   "mirror")
            for i in range(0, 9):
                print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
            print_line += '{:>12} \\\\'.format(int(cnt))
            print(print_line)
            if self.logger:
                self.logger.info(print_line)

            # print non-mirror area score
            print_line = "{:>12}& {:>12}& {:>18}& {:>12}& ".format(self.input_tag, self.Train_tag, self.method_tag,
                                                                   "non-mirror")
            for i in range(9, 18):
                print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
            print_line += '{:>12} \\\\'.format(int(cnt))
            print(print_line)
            if self.logger:
                self.logger.info(print_line)

            # print all area score
            print_line = "{:>12}& {:>12}& {:>18}& {:>12}& ".format(self.input_tag, self.Train_tag, self.method_tag,
                                                                   "all")
            for i in range(18, 27):
                print_line += '{:>12.3f}& '.format(eval_measures_cpu[i])
            print_line += '{:>12} \\\\'.format(int(cnt))
            print(print_line)
            if self.logger:
                self.logger.info(print_line)
            print("result saved to : ", self.main_output_folder)

        if self.logger:
            self.logger.info(
                "######################################## {:>20} ########################################".format(
                    "compared with refD"))
        print("######################################## {:>20} ########################################".format(
            "compared with refD"))
        print_all_scores(self.m_nm_all_refD / self.ref_cnt, self.ref_cnt)
        self.save_as_table_format(self.m_nm_all_refD / self.ref_cnt, compare_with_raw=False)
        if self.logger:
            self.logger.info(
                "######################################## {:>20} ########################################".format(
                    "compared with rawD"))
        print("######################################## {:>20} ########################################".format(
            "compared with rawD"))
        print_all_scores(self.m_nm_all_rawD / self.raw_cnt, self.raw_cnt)
        self.save_as_table_format(self.m_nm_all_rawD / self.raw_cnt, compare_with_raw=True)

        if self.cal_std:
            self.cal_std_for_all(self.m_nm_all_refD / self.ref_cnt, compare_with_raw=False)
            self.cal_std_for_all(self.m_nm_all_rawD / self.raw_cnt, compare_with_raw=True)

        self.save_sampleScore(self.main_output_folder)

    def save_sampleScore(self, method_output_folder):
        one_output_path = os.path.join(method_output_folder,
                                       "minFilter_{}_full_{}_score_per_sample.json".format(self.min_threshold_filter,
                                                                                           self.get_full_set))
        save_json(one_output_path, self.sample_score)

    def cal_std_for_all(self, avg_score, compare_with_raw):
        eval_measures_std = []
        for one_score_index, one_score in enumerate(avg_score):
            scores = []
            for item in self.sample_score.items():
                try:
                    if compare_with_raw:
                        scores.append(item[1]["raw"][one_score_index])
                    else:
                        scores.append(item[1]["ref"][one_score_index])
                except:
                    continue
            eval_measures_std.append(np.std(scores) / np.sqrt(len(scores)))
        self.save_as_table_format(eval_measures_std, compare_with_raw=compare_with_raw, compute_std=True)

    def compute_and_update_mirror3D_metrics(self, pred_depth, depth_shift, color_image_path, rawD_gt_depth_path,
                                            refD_gt_depth_path, mask_path):
        if color_image_path.find("mp3d") > 0 and "mesh" not in self.Train_tag:
            self.Train_tag = self.Train_tag.replace("ref", "mesh-ref")
            self.Train_tag = self.Train_tag.replace("raw", "mesh")

        def compute_errors(gt, pred, eval_area):  # ! gt and pred are in m

            gt = np.array(gt, dtype="f")
            pred = np.array(pred, dtype="f")

            min_depth_eval = 1e-3
            max_depth_eval = 10

            pred[pred < min_depth_eval] = min_depth_eval
            pred[np.isinf(pred)] = max_depth_eval

            gt[np.isinf(gt)] = 0
            gt[np.isnan(gt)] = 0

            if self.min_threshold_filter:
                valid_mask = gt > min_depth_eval  # np.logical_and(gt > min_depth_eval)#, gt < max_depth_eval
            else:
                valid_mask = np.ones(gt.shape).astype(bool)

            scale = np.sum(pred[valid_mask] * gt[valid_mask]) / np.sum(pred[valid_mask] ** 2)
            valid_mask = np.logical_and(valid_mask, eval_area)

            ssim_obj = SSIM()
            ssim_map = ssim_obj.forward(torch.tensor(pred * valid_mask.astype(int)).unsqueeze(0).unsqueeze(0),
                                        torch.tensor(gt * valid_mask.astype(int)).unsqueeze(0).unsqueeze(0))
            ssim = ssim_map[valid_mask].mean()

            gt = gt[valid_mask]
            pred = pred[valid_mask]

            if valid_mask.sum() == 0 or sum(gt) == 0:
                return np.array(False)

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

            scaled_rms = np.sqrt(((scale * pred - gt) ** 2).mean())
            return rmse, scaled_rms, rel, ssim.item(), d105, d110, d125, d125_2, d125_3

        def get_refD_scores(pred_depth, depth_shift, color_image_path):
            if not os.path.exists(mask_path) and "refined" in refD_gt_depth_path:
                return

            depth_shift = np.array(depth_shift)
            ref_d_gt_depth = cv2.resize(cv2.imread(refD_gt_depth_path, cv2.IMREAD_ANYDEPTH),
                                        (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
            ref_d_gt_depth = np.array(ref_d_gt_depth) / depth_shift
            pred_depth = np.array(pred_depth)

            pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
            ref_d_gt_depth = cv2.resize(ref_d_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)

            if "no_mirror" in color_image_path:
                mirror_error = tuple([0, 0, 0, 0, 1, 1, 1, 1, 1])
                all_image_error = compute_errors(ref_d_gt_depth, pred_depth, True)
                non_mirror_error = all_image_error
            else:
                mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0,
                                         cv2.INTER_NEAREST)
                mirror_error = compute_errors(ref_d_gt_depth, pred_depth, mirror_mask > 0)
                non_mirror_error = compute_errors(ref_d_gt_depth, pred_depth, mirror_mask == False)
                all_image_error = compute_errors(ref_d_gt_depth, pred_depth, True)

            if not all_image_error or not mirror_error or not non_mirror_error:
                return

            one_m_nm_all = mirror_error + non_mirror_error + all_image_error
            return one_m_nm_all

        def get_rawD_scores(pred_depth, depth_shift, color_image_path):
            if not os.path.exists(mask_path):
                return
            depth_shift = np.array(depth_shift)
            ref_d_gt_depth = cv2.resize(cv2.imread(rawD_gt_depth_path, cv2.IMREAD_ANYDEPTH),
                                        (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST)
            ref_d_gt_depth = np.array(ref_d_gt_depth) / depth_shift
            pred_depth = np.array(pred_depth)

            pred_depth = cv2.resize(pred_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)
            ref_d_gt_depth = cv2.resize(ref_d_gt_depth, (self.width, self.height), 0, 0, cv2.INTER_NEAREST)

            if "no_mirror" in color_image_path:
                mirror_error = tuple([0, 0, 0, 0, 1, 1, 1, 1, 1])
                all_image_error = compute_errors(ref_d_gt_depth, pred_depth, True)
                non_mirror_error = all_image_error
            else:
                mirror_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height), 0, 0,
                                         cv2.INTER_NEAREST)
                mirror_error = compute_errors(ref_d_gt_depth, pred_depth, mirror_mask > 0)
                non_mirror_error = compute_errors(ref_d_gt_depth, pred_depth, mirror_mask == False)
                all_image_error = compute_errors(ref_d_gt_depth, pred_depth, True)

            if not all_image_error or not mirror_error or not non_mirror_error:
                return
            one_m_nm_all = mirror_error + non_mirror_error + all_image_error
            return one_m_nm_all

        one_ref_m_nm_all = []
        one_raw_m_nm_all = []
        try:
            one_ref_m_nm_all = torch.tensor(get_refD_scores(np.array(pred_depth).copy(), depth_shift, color_image_path))
            self.m_nm_all_refD += one_ref_m_nm_all
            self.ref_cnt += 1
            one_ref_m_nm_all = one_ref_m_nm_all.tolist()
        except:
            print(color_image_path, "can't calculate ref error")

        try:
            one_raw_m_nm_all = torch.tensor(get_rawD_scores(np.array(pred_depth).copy(), depth_shift, color_image_path))
            self.m_nm_all_rawD += one_raw_m_nm_all
            self.raw_cnt += 1
            one_raw_m_nm_all = one_raw_m_nm_all.tolist()
        except:
            print(color_image_path, "can't calculate raw error")
        if self.save_score_per_sample:
            img_name = color_image_path.split("/")[-1]
            self.sample_score[img_name] = {"ref": one_ref_m_nm_all, "raw": one_raw_m_nm_all}

        return

    def save_result(self, main_output_folder, pred_depth, depth_shift, color_img_path, rawD_gt_depth_path,
                    refD_gt_depth_path, mask_path):

        self.main_output_folder = main_output_folder
        if os.path.exists(self.main_output_folder) and self.to_create_folder:
            self.main_output_folder = self.main_output_folder + "_{}".format(random.randint(0, 10000))
            os.makedirs(self.main_output_folder, exist_ok=True)
            self.to_create_folder = False
        elif not os.path.exists(self.main_output_folder):
            self.to_create_folder = False

        pred_depth = np.array(pred_depth)
        depth_shift = np.array(depth_shift)

        info_txt_save_path = os.path.join(main_output_folder, "color_mask_gtD_predD.txt")
        pred_depth_scaled = pred_depth * depth_shift
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        raw_d_folder = os.path.join(self.dataset_root, rawD_gt_depth_path.replace(self.dataset_root, "").split("/")[1])
        depth_np_save_path = rawD_gt_depth_path.replace(raw_d_folder, main_output_folder + "/pred_depth")
        pred_d_save_folder = os.path.split(depth_np_save_path)[0]
        os.makedirs(pred_d_save_folder, exist_ok=True)
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
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
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

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

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


class MirrorSegEval:
    def __init__(self, width=640, height=480):
        self.f_measure_list = []
        self.IOU_list = []
        self.MAE_list = []
        self.width = width
        self.height = height

    def reset_setting(self, width=640, height=480):
        self.f_measure_list = []
        self.IOU_list = []
        self.MAE_list = []
        self.width = width
        self.height = height

    def print_seg_score(self):
        print("|{:<25} : {:<5}|".format("evaluation sample num", len(self.f_measure_list)))
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
