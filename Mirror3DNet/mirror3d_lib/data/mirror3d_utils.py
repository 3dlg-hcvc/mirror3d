import numpy as np
import pycocotools.mask as mask_util

from planrcnn_detectron2_lib.structures import BoxMode

from detectron2.data import transforms as T

def transform_instance_annotations( # chris : transform on annotation item
    annotation, transforms, image_size, *, keypoint_hflip_indices=None, anchor_normals=None,
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints
    
    if "anchor_normal_class" in annotation and not sum(isinstance(t, T.NoOpTransform) for t in transforms.transforms):
        anchor_normal_class, anchor_normal_residual = transfor_anchor_annotation(anchor_normals, annotation["mirror_normal_camera"],transforms)
        annotation["anchor_normal_class"] = anchor_normal_class
        annotation["anchor_normal_residual"] = anchor_normal_residual

    return annotation


def transfor_anchor_annotation(anchor_normals, mirror_normal_camera, transforms):

    hor_flip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms)
    ver_flip = sum(isinstance(t, T.VFlipTransform) for t in transforms.transforms)

    if hor_flip: # x = -x
        mirror_normal_camera[0] = -mirror_normal_camera[0]
    elif ver_flip:
        mirror_normal_camera[1] = -mirror_normal_camera[1]

    cloest_distance = 100 # init to be a large number
    for i in range(len(anchor_normals)):
        distance_anchor = mirror_normal_camera - anchor_normals[i]
        distance = np.sqrt(distance_anchor[0]**2 + distance_anchor[1]**2 + distance_anchor[2]**2)
        if distance < cloest_distance:
            cloest_distance = distance
            anchor_normal_class = i #! the last class is background
            anchor_normal_residual = distance_anchor
    
    return anchor_normal_class, list(anchor_normal_residual)

