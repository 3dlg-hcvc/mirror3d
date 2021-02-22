BTS refined_D result : # TODO /local-scratch/jiaqit/exp/chris_mirror_proj/bts/pytorch/important_model/bts_m3d_2020-10-05-14-31-33/model-2500-best_rms_0.47589 
######### input ######### /project/3dlg-hcvc/jiaqit/waste/2020-11-10-22-49-57/bts_rawD_only_mirror_mask_gtDepth_pDepth_npDepth.txt
Namespace(debug=False, depth_shift=4000, height=256, m='bts', resize=True, txt='/project/3dlg-hcvc/jiaqit/waste/2020-11-10-22-49-57/bts_rawD_only_mirror_mask_gtDepth_pDepth_npDepth.txt', whole_img=False, width=320)
################### Evaluation : refined depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.958 &     0.558 &     0.478 &     0.105 &     0.203 &     0.452 &     0.715 &     0.849 &     0.868 &     0.557 &     0.478 &     0.799 &     0.388 &    13.884 &     0.156 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.791 &     0.596 &     0.351 &     0.124 &     0.236 &     0.492 &     0.769 &     0.895 &     0.636 &     0.557 &     0.351 &     0.383 &     0.370 &    25.440 &     0.133 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.886 &     0.599 &     0.405 &     0.118 &     0.224 &     0.469 &     0.740 &     0.873 &     0.700 &     0.557 &     0.405 &     0.543 &     0.401 &    27.843 &     0.144 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.028 &     0.629 &     0.516 &     0.099 &     0.186 &     0.434 &     0.700 &     0.828 &     0.915 &     0.556 &     0.516 &     0.899 &     0.415 &    16.768 &     0.164 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.791 &     0.595 &     0.351 &     0.124 &     0.236 &     0.492 &     0.769 &     0.895 &     0.636 &     0.558 &     0.351 &     0.383 &     0.370 &    25.440 &     0.133 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.892 &     0.610 &     0.412 &     0.118 &     0.224 &     0.469 &     0.738 &     0.870 &     0.704 &     0.558 &     0.412 &     0.561 &     0.406 &    28.198 &     0.145 \\


BTS raw_D result : # TODO /local-scratch/jiaqit/exp/chris_mirror_proj/bts/pytorch/important_model/bts_m3d_noisy_2020-10-06-18-14-15/model-16500-best_rms_0.56362
######### input ######### /project/3dlg-hcvc/jiaqit/waste/2020-11-10-22-56-45/bts_rawD3k_only_mirror_mask_gtDepth_pDepth_npDepth.txt
Namespace(debug=False, depth_shift=4000, height=256, m='bts', resize=True, txt='/project/3dlg-hcvc/jiaqit/waste/2020-11-10-22-56-45/bts_rawD3k_only_mirror_mask_gtDepth_pDepth_npDepth.txt', whole_img=False, width=320)
################### Evaluation : refined depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.792 &     1.420 &     0.881 &     0.067 &     0.128 &     0.279 &     0.490 &     0.663 &     1.547 &     0.564 &     0.881 &     2.202 &     0.599 &    27.050 &     0.236 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.798 &     0.658 &     0.359 &     0.120 &     0.231 &     0.496 &     0.773 &     0.896 &     0.637 &     0.564 &     0.359 &     0.402 &     0.371 &    26.138 &     0.133 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.043 &     0.729 &     0.470 &     0.109 &     0.209 &     0.453 &     0.716 &     0.846 &     0.780 &     0.564 &     0.470 &     0.750 &     0.441 &    32.236 &     0.155 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.753 &     1.384 &     0.892 &     0.068 &     0.133 &     0.295 &     0.503 &     0.671 &     1.502 &     0.567 &     0.892 &     2.223 &     0.599 &    28.020 &     0.234 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.798 &     0.651 &     0.359 &     0.120 &     0.231 &     0.496 &     0.773 &     0.896 &     0.637 &     0.569 &     0.359 &     0.402 &     0.371 &    26.138 &     0.133 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.034 &     0.723 &     0.473 &     0.110 &     0.211 &     0.455 &     0.718 &     0.847 &     0.776 &     0.569 &     0.473 &     0.763 &     0.442 &    32.195 &     0.155 \\



saic refineD result : # TODO /local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/logs/LRN|CRP|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001/snapshots2020-10-29-10-37-32/snapshot_160.pth
/local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/logs/LRN|CRP|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001/snapshots2020-10-29-10-37-32/snapshot_160.pth
######### input ######### /project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-08-19/gt_pred_mask_raw_list.txt
Namespace(debug=False, depth_shift=4000, height=256, m='saic', resize=True, txt='/project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-08-19/gt_pred_mask_raw_list.txt', whole_img=False, width=320)
################### Evaluation : refined depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.517 &     0.413 &     0.199 &     0.407 &     0.606 &     0.797 &     0.902 &     0.947 &     0.422 &     0.791 &     0.199 &     0.293 &     0.196 &    12.284 &     0.070 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.398 &     0.413 &     0.106 &     0.814 &     0.871 &     0.922 &     0.954 &     0.970 &     0.190 &     0.791 &     0.106 &     0.157 &     0.154 &    14.055 &     0.033 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.498 &     0.437 &     0.148 &     0.735 &     0.808 &     0.881 &     0.929 &     0.955 &     0.253 &     0.791 &     0.148 &     0.237 &     0.198 &    17.740 &     0.046 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.588 &     0.483 &     0.232 &     0.391 &     0.571 &     0.766 &     0.882 &     0.930 &     0.462 &     0.787 &     0.232 &     0.361 &     0.227 &    14.868 &     0.079 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.398 &     0.409 &     0.106 &     0.814 &     0.871 &     0.922 &     0.954 &     0.970 &     0.190 &     0.789 &     0.106 &     0.157 &     0.154 &    14.055 &     0.033 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.505 &     0.447 &     0.153 &     0.739 &     0.809 &     0.879 &     0.927 &     0.953 &     0.256 &     0.789 &     0.153 &     0.250 &     0.201 &    18.037 &     0.046 \\

# ! 200 DE  /project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-34-12/gt_pred_mask_raw_list.txt

######### input ######### /project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-34-12/gt_pred_mask_raw_list.txt
Namespace(debug=False, depth_shift=4000, height=256, m='saic', resize=True, txt='/project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-34-12/gt_pred_mask_raw_list.txt', whole_img=False, width=320)
################### Evaluation : refined depth ###################
#### number of mirror sample #### 
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.306 &     1.205 &     0.139 &     0.299 &     0.486 &     0.794 &     0.916 &     0.958 &     0.883 &     0.459 &     0.139 &     0.247 &     0.247 &    22.089 &     0.070 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.311 &     1.208 &     0.139 &     0.299 &     0.485 &     0.789 &     0.916 &     0.958 &     0.887 &     0.450 &     0.139 &     0.249 &     0.248 &    22.124 &     0.070 \\


saic rawD result : # TODO /local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/logs/LRN|CRP|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001/snapshots2020-10-29-19-12-44/snapshot_160.pth
/local-scratch/jiaqit/exp/chris_mirror_proj/saic_depth_completion/logs/LRN|CRP|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001/snapshots2020-10-29-19-12-44/snapshot_160.pth
/project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-14-18/gt_pred_mask_raw_list.txt


Namespace(debug=False, depth_shift=4000, height=256, m='saic', resize=True, txt='/project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-14-18/gt_pred_mask_raw_list.txt', whole_img=False, width=320)
################### Evaluation : refined depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.902 &     1.431 &     0.798 &     0.150 &     0.201 &     0.308 &     0.474 &     0.622 &     1.565 &     0.754 &     0.798 &     2.108 &     0.634 &    35.500 &     0.237 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.494 &     0.595 &     0.133 &     0.861 &     0.884 &     0.908 &     0.932 &     0.950 &     0.201 &     0.754 &     0.133 &     0.230 &     0.205 &    18.727 &     0.039 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.864 &     0.696 &     0.266 &     0.754 &     0.780 &     0.815 &     0.859 &     0.894 &     0.386 &     0.754 &     0.266 &     0.593 &     0.335 &    29.794 &     0.072 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 484
--------------------mirror area         --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.877 &     1.436 &     0.786 &     0.191 &     0.239 &     0.333 &     0.492 &     0.632 &     1.519 &     0.756 &     0.786 &     2.082 &     0.627 &    35.883 &     0.230 \\
--------------------none-mirror area    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.494 &     0.585 &     0.133 &     0.861 &     0.884 &     0.908 &     0.932 &     0.950 &     0.201 &     0.758 &     0.133 &     0.230 &     0.205 &    18.727 &     0.039 \\
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    0.849 &     0.690 &     0.262 &     0.764 &     0.789 &     0.821 &     0.863 &     0.897 &     0.376 &     0.758 &     0.262 &     0.587 &     0.330 &    29.259 &     0.070 \\

# ! 200 DE /project/3dlg-hcvc/jiaqit/waste/2020-11-10-23-31-18/gt_pred_mask_raw_list.txt 
--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.908 &     1.840 &     0.177 &     0.629 &     0.666 &     0.717 &     0.765 &     0.822 &     1.073 &     0.582 &     0.177 &     0.561 &     0.468 &    40.241 &     0.111 \\
################### Evaluation : raw depth ###################
#### number of mirror sample #### 1

--------------------all mirror image    --------------------
      RMSE,     s-RMSE,        Rel,     d_1.05,     d_1.10,     d_1.25,   d_1.25^2,   d_1.25^3,        MAE,       SSIM,     AbsRel,      SqRel,    RMSElog,      SILog,      log10
    1.909 &     1.841 &     0.177 &     0.629 &     0.667 &     0.718 &     0.766 &     0.823 &     1.071 &     0.581 &     0.177 &     0.560 &     0.467 &    40.211 &     0.110 \\












bts ft refineD
output
/project/3dlg-hcvc/jiaqit/waste/2020-11-11-17-33-50/bts_rawD3k_only_mirror_mask_gtDepth_pDepth_npDepth.txt