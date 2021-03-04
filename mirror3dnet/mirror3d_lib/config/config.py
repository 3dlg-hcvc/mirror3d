from detectron2.config import CfgNode

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from detectron2.config.defaults import _C
    _C.DEPTH_EST = True
    _C.ANCHOR_REG = True
    _C.ANCHOR_CLS = True
    _C.OBJECT_CLS = True
    _C.OBJECT_SEG = True
    _C.OBJECT_BBOX_REG = True
    _C.RGBD_INPUT = False

    # ---- DEPTH_EST -----
    _C.DEPTH_SHIFT = 1000.0
    _C.FOCAL_LENGTH = 320
    _C.INVERSE_DEPTH = False
    _C.REFINED_DEPTH = True
    _C.MESH_DEPTH = True

    # ---- ANCHOR_REG -----
    _C.ANCHOR_REG_METHOD = 1
    _C.UNIT_ANCHOR_NORMAL = True

    # ---- ANCHOR_CLS -----
    _C.ANCHOR_NORMAL_NYP = ""

    ############## eval configure ##############
    _C.EVAL = False

    _C.EVAL_ANCHOR_AP = True
    _C.EVAL_CLS_AP = True
    # normal l2 loss and normal angle difference
    _C.EVAL_MASK_IOU = True
    _C.EVAL_NORMAL = True
    _C.EVAL_BRANCH_ORI_DEPTH = True
    _C.EVAL_BRANCH_REF_DEPTH = True
    _C.EVAL_INPUT_REF_DEPTH = True
    _C.EVAL_HEIGHT = 480
    _C.EVAL_WIDTH = 640

    # ---- eval output ----
    _C.EVAL_SAVE_MASKED_IMG = True
    # if only train depth branch save the output depth | if refine save the refinedD
    _C.EVAL_SAVE_DEPTH = True
    _C.EVAL_SAVE_NORMAL_VIS = True
    _C.MASK_ON_OTHER_SIZE = False

    # --------- refine depth -----------
    _C.REF_DEPTH_TO_REFINE = ""
    # " rawD_mirror / rawD_border / DE_mirror / DE_border " 
    _C.REF_MODE = "rawD_border"
    _C.REF_BORDER_WIDTH = 50

    _C.ANCHOR_NORMAL_CLASS_NUM = 8 # 
    _C.TRAIN_COCO_JSON = ()
    _C.VAL_COCO_JSON = ()
    _C.TRAIN_IMG_ROOT = ()
    _C.VAL_IMG_ROOT = ()
    _C.TRAIN_NAME = ()
    _C.VAL_NAME = ()

    return _C.clone()