__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools.cocoeval import COCOeval, maskUtils
import copy
import numpy as np
import math

class Mirror3dCOCOeval(COCOeval):
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', cfg=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.anchor_evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self._anchor_gts = defaultdict(list)       # gt for evaluation
        self._anchor_dts = defaultdict(list)       # dt for evaluation
        self.params = Planercnn_Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds()) # one class : 1
        # self.params.catIds = [0,1]

        self.params.anchor_catIds = [i for i in range(cfg.ANCHOR_NORMAL_CLASS_NUM)]
        self.anchor_normals = np.load(cfg.ANCHOR_NORMAL_NYP)
        if cfg.UNIT_ANCHOR_NORMAL:
            for i in range(len(self.anchor_normals)):
                self.anchor_normals[i] = self.anchor_normals[i]/ (np.sqrt(self.anchor_normals[i][0]**2 + self.anchor_normals[i][1]**2 + self.anchor_normals[i][2]**2))
        self.cfg = cfg

    
    

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation

        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.anchor_evalImgs = defaultdict(list)  
        self.eval     = {}                  # accumulated evaluation results

        #### ----- chris : load anchor gt ----- ####
        if self.cfg.ANCHOR_CLS:
            if p.useCats:
                anchor_gts=self.cocoGt.loadAnns(self.cocoGt.get_achorAnnIds(imgIds=p.imgIds, catIds=p.anchor_catIds))
                anchor_dts=self.cocoDt.loadAnns(self.cocoDt.get_achorAnnIds(imgIds=p.imgIds, catIds=p.anchor_catIds))
            else:
                anchor_gts=self.cocoGt.loadAnns(self.cocoGt.get_achorAnnIds(imgIds=p.imgIds))
                anchor_dts=self.cocoDt.loadAnns(self.cocoDt.get_achorAnnIds(imgIds=p.imgIds))

            # convert ground truth to mask if iouType == 'segm'
            if p.iouType == 'segm':
                _toMask(anchor_gts, self.cocoGt)
                _toMask(anchor_dts, self.cocoDt)
            # set ignore flag
            for gt in anchor_gts:
                gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
                if p.iouType == 'keypoints':
                    gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
            self._anchor_gts = defaultdict(list)       # gt for evaluation
            self._anchor_dts = defaultdict(list)       # dt for evaluation

            for gt in anchor_gts:
                self._anchor_gts[gt['image_id'], gt['anchor_normal_class']].append(gt)
            for dt in anchor_dts:
                self._anchor_dts[dt['image_id'], dt['anchor_normal_class']].append(dt)

    

        

    # chris : get normal_anchor specific score
    def evaluate_anchor_ap(self):

        
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number

        if p.iouType == 'segm' or p.iouType == 'bbox':
            anchor_specific_computeIoU = self.anchor_specific_computeIoU
        elif p.iouType == 'keypoints':
            anchor_specific_computeIoU = self.computeOks
        # ious between all gts and dts : self.ious = img_ID : {anchor_specific_computeIoU}
        self.ious = {(imgId, catId): anchor_specific_computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in p.anchor_catIds}

        evaluateImg = self.anchor_evaluateImg
        maxDet = p.maxDets[-1] # 100  
        self.anchor_evalImgs = [evaluateImg(imgId, anchor_catId, areaRng, maxDet) # thresholds on max detections per image
                 for anchor_catId in p.anchor_catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]

        self._paramsEval = copy.deepcopy(self.params)

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def evaluate_normal(self, cfg):

        # TODO img_save_folder = ？？？

        import random
        import matplotlib.pyplot as plt
        import os
        import time
        import cv2
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)
        
        def cos_sim(vector_a, vector_b):
            vector_a = np.mat(vector_a)
            vector_b = np.mat(vector_b)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            if np.isnan(sim):
                sim = 1
            return sim

        def dotproduct(v1, v2):
            return sum((a*b) for a, b in zip(v1, v2))

        def length(v):
            return math.sqrt(dotproduct(v, v))

        def angle(v1, v2):
            return ((math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))/np.pi)*180

        def save_txt(save_path, data):
            with open(save_path, "w") as file:
                for info in data:
                    file.write(str(info))
                    file.write("\n")
            print("-------------------------------- txt saved to : ", save_path, len(data))
        
        def read_txt(txt_path):
            with open(txt_path, "r") as file:
                lines = file.readlines()
            return [line.strip() for line in lines]
        
        def draw_bbox(pred_bboxes, gt_bboxes, image_path, img_save_folder, improve, distance):
            success_save_folder = os.path.join(img_save_folder, "success")
            fail_save_folder = os.path.join(img_save_folder, "fail")
            os.makedirs(success_save_folder, exist_ok=True)
            os.makedirs(fail_save_folder, exist_ok=True)
            img = cv2.imread(image_path)
            cv2.rectangle(img, (int(gt_bboxes[0]),int(gt_bboxes[1])), (int(gt_bboxes[0]) + int(gt_bboxes[2]),int(gt_bboxes[1]) + int(gt_bboxes[3])), (0,255,0), 3)
            cv2.rectangle(img, (int(pred_bboxes[0]),int(pred_bboxes[1])), (int(pred_bboxes[0]) + int(pred_bboxes[2]),int(pred_bboxes[1]) + int(pred_bboxes[3])), (0,0,255), 3)
            
            if improve >= 0:
                cv2.putText(img, "dis {:0.2f} +{:0.2f}".format(distance, improve), (40,40), 1, 3, (0,255,0), 2)
                img_save_name = "{:0.2f}_success_bbox_vis_".format(improve) + image_path.split("/")[-3] + "_" + image_path.split("/")[-1]
                img_save_path = os.path.join(success_save_folder, img_save_name)
            else:
                cv2.putText(img, "dis {:0.2f} {:0.2f}".format(distance, improve), (40,40), 1, 3, (0,255,0), 2)
                img_save_name = "{:0.2f}_fail_bbox_vis_".format(improve) + image_path.split("/")[-3] + "_" + image_path.split("/")[-1]
                img_save_path = os.path.join(fail_save_folder, img_save_name)
            cv2.imwrite(img_save_path, img)
            print("bbox_vis image saved to : ",  img_save_path, "distance {:0.4f} improve {:0.4f}".format(distance, improve))

        def print_score(score_tag, iou_predNum_noneNum, iou_scoreList):
            IOU_list = []
            for iou in self.params.iouThrs:
                if len(iou_predNum_noneNum[iou][0]):
                    IOU_list.append([np.array(iou_scoreList[iou])[:,0].mean(), np.array(iou_scoreList[iou])[:,1].mean(), np.array(iou_scoreList[iou])[:,2].mean()])

            print("| {:45} | IOU 0.5:0.95 | {:15} {:5f} |".format("mirror_pred_anchor & mirror_GT_normal", score_tag, np.array(IOU_list)[:,0].mean()))
            print("| {:45} | IOU 0.5:0.95 | {:15} {:5f} |".format("mirror_pred_normal & mirror_GT_normal", score_tag, np.array(IOU_list)[:,1].mean()))
            print("| {:45} | IOU 0.5:0.95 | {:15} {:5f} |".format("mirror_pred_res & mirror_GT_res", score_tag, np.array(IOU_list)[:,2].mean()))

        def eval_normal():
            all_areaRng = None
            for areaRng, areaRngLbl in zip(self.params.areaRng, self.params.areaRngLbl):
                if areaRngLbl == "all":
                    all_areaRng = areaRng
                    break

            iou_l2lossList = dict()
            for iou in self.params.iouThrs:
                iou_l2lossList[iou] = []

            iou_CSlist = dict()
            for iou in self.params.iouThrs:
                iou_CSlist[iou] = []
            
            iou_angle_list = dict()
            for iou in self.params.iouThrs:
                iou_angle_list[iou] = []

            iou_predNum_noneNum = dict()
            for iou in self.params.iouThrs:
                iou_predNum_noneNum[iou] = [set(),set()]
            
        
            for one_eval_img in self.anchor_evalImgs: #  [evaluateImg(imgId (81), anchor_catId (anchor_normal_num + bg), areaRng (4 : s m l all), maxDet (100))
                if one_eval_img: # if one_eval_img is not None
                    if one_eval_img["aRng"] == all_areaRng: # only care about the performace on "All" image prediction
                        for iou_idx, iou in enumerate(self.params.iouThrs):
                            image_id = one_eval_img["image_id"]
                            anchor_cat_id = one_eval_img["category_id"]
                            gt_anchor_id = one_eval_img["dtMatches"][iou_idx] # GT index that dt match
                            dt_anchor_id = one_eval_img["gtMatches"][iou_idx] # dt index that GT match
                            
                            zero_count = 0
                            for idx, one_match_gt_id in enumerate(gt_anchor_id):
                                if one_match_gt_id > 0 : # one_match_gt_id = 0 then this instance match background (gt[0] is background)
                                    # get the matched GT instance 

                                    for gt_item in self._anchor_gts[image_id, anchor_cat_id]:
                                        if gt_item["id"] == one_match_gt_id:
                                            mirror_GT_normal = unit_vector(self.anchor_normals[gt_item["anchor_normal_class"]] + np.array(gt_item["anchor_normal_residual"])) # GT mirror normal 
                                            mirror_GT_res = np.array(gt_item["anchor_normal_residual"])
                                            mirror_GT_anchor = self.anchor_normals[gt_item["anchor_normal_class"]]
                                            mirror_GT_normal_class = gt_item["anchor_normal_class"]
                                            mirror_GT_bboxes = gt_item["bbox"]
                                            mirror_image_path = gt_item["image_path"]
                                            break

                                    # get the matched predicted instance 
                                    have_match = False
                                    for pred_item in self._anchor_dts[image_id, anchor_cat_id]:
                                        if  pred_item["id"] == dt_anchor_id[idx - zero_count]:
                                            mirror_pred_normal = unit_vector(self.anchor_normals[pred_item["anchor_normal_class"]] + np.array(pred_item["anchor_normal_residual"])) #  pred mirror normal
                                            mirror_pred_res = np.array(pred_item["anchor_normal_residual"])
                                            mirror_pred_anchor = self.anchor_normals[pred_item["anchor_normal_class"]]
                                            mirror_pred_normal_class = pred_item["anchor_normal_class"]
                                            mirror_pred_bboxes = pred_item["bbox"]
                                            have_match = True
                                            break
                                    try:
                                        improve = np.linalg.norm(mirror_pred_anchor - mirror_GT_normal) - np.linalg.norm(mirror_pred_normal - mirror_GT_normal)
                                        distance = np.linalg.norm(mirror_GT_normal - mirror_GT_anchor)
                                    except Exception as e:
                                        improve = 0
                                        distance = 0
                                        print(e)
                                        print("error ------------ one_match_gt_id, gt_anchor_id : ", one_match_gt_id, gt_anchor_id)
                                        mirror_pred_anchor = np.array([0,0,0])
                                        mirror_GT_normal = np.array([0,0,0])
                                        mirror_pred_normal = np.array([0,0,0])
                                        mirror_GT_anchor = np.array([0,0,0])
                                        mirror_GT_res = np.array([0,0,0])
                                        mirror_pred_res = np.array([0,0,0])
                                        print(dt_anchor_id[idx - zero_count], dt_anchor_id)
                                        
                                    if not have_match:
                                        iou_predNum_noneNum[iou][0].add(image_id)
                                        continue
                                    #################################### get L2 loss ####################################  
                                    iou_l2lossList[iou].append([np.linalg.norm(mirror_pred_anchor - mirror_GT_normal),\
                                                            np.linalg.norm(mirror_pred_normal - mirror_GT_normal), \
                                                            np.linalg.norm(mirror_pred_res - mirror_GT_res),\
                                                            improve,\
                                                            distance])
                                    #################################### get cos similarity ####################################
                                    iou_CSlist[iou].append([cos_sim(mirror_pred_anchor, mirror_GT_normal),\
                                                            cos_sim(mirror_pred_normal, mirror_GT_normal), \
                                                            cos_sim(mirror_pred_res, mirror_GT_res)])
                                    iou_angle_list[iou].append([angle(mirror_pred_anchor, mirror_GT_normal),\
                                                            angle(mirror_pred_normal, mirror_GT_normal), \
                                                            angle(mirror_pred_res, mirror_GT_res)])
                                    # print(iou_angle_list[iou])
                                    iou_predNum_noneNum[iou][0].add(image_id)
                                
                                # if iou_idx == 0 and cfg.EVAL_RES_BBOX:
                                #     bbox_vis_save_folder = os.path.join(img_save_folder, "bbox_vis")
                                #     os.makedirs(bbox_vis_save_folder, exist_ok=True)
                                #     draw_bbox(mirror_pred_bboxes, mirror_GT_bboxes, mirror_image_path, bbox_vis_save_folder, improve, distance)
                                else:
                                    # iou_CSlist[iou].append([0,0,0])
                                    # iou_angle_list[iou].append([180,180,180])
                                    zero_count += 1
                                    iou_predNum_noneNum[iou][1].add(image_id)
            

            #################################### print prediction situation ####################################  
            for iou in self.params.iouThrs:
                print("| IOU {:.2f} : {:5}/{:5} have prediction|".format(iou, len(iou_predNum_noneNum[iou][0]), len( iou_predNum_noneNum[iou][0].union(iou_predNum_noneNum[iou][1]) )))
            
            if len(iou_predNum_noneNum[self.params.iouThrs[0]][0]) == 0:
                return


            print_score("L2_loss", iou_predNum_noneNum, iou_l2lossList)
            print_score("SSIM", iou_predNum_noneNum, iou_CSlist)
            print_score("angle_diff", iou_predNum_noneNum, iou_angle_list)
            
            #################################### get cos similarity ####################################

    # chris : compute IOU based on self._anchor_gts & self._anchor_dts
    def anchor_specific_computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._anchor_gts[imgId,catId]
            dt = self._anchor_dts[imgId,catId] # self._anchor_dts[dt['image_id'], dt['anchor_normal_class']].append(dt)
        else:
            gt = [_ for cId in p.catIds for _ in self._anchor_gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._anchor_dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        # ! sort pred_instance by anchor_score
        inds = np.argsort([-d['anchor_score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def anchor_accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.anchor_evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.anchor_catIds = p.anchor_catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.anchor_catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        anchor_catIds = _pe.anchor_catIds if _pe.useCats else [-1]
        setK = set(anchor_catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.anchor_catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.anchor_evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def anchor_evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._anchor_gts[imgId,catId] 
            dt = self._anchor_dts[imgId,catId]
        else:
            gt = [_ for cId in p.anchor_catIds for _ in self._anchor_gts[imgId,cId]]
            dt = [_ for cId in p.anchor_catIds for _ in self._anchor_dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['anchor_score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId] # ious between all gts and dts

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category

        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt], # chris : stores all the predict_instance_id if the predict_instance category the same as catID;
                'gtIds':        [g['id'] for g in gt], # chris : stores all the GT_instance_id if the GT_instance category the same as catID; cloud be [] if there is no gt_instance in the image of class catID
                'dtMatches':    dtm, # if 'gtIds' = [] this must be ([[0],[0]...])
                'gtMatches':    gtm, # if 'gtIds' = [] this must be blank
                'dtScores':     [d['anchor_score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def chris_summarize(self, tag):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        title = tag + "_AP"
        print("| {:45} | IOU 0.5:0.95 |  {:5f} |".format( title, _summarize(1)))


class Planercnn_Params:
    '''
    Planercnn_Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        self.anchor_catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        self.anchor_catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
