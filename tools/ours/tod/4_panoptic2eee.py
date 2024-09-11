#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data
The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from detectron2.data import detection_utils as utils
import pycocotools.mask as mask_util
import cv2
from panopticapi.utils import rgb2id
from tqdm import tqdm
from pycocotools import mask as m


def masks_to_fg_mask(masks):
    # masks = [N, H, W], numpy array
    fg_mask = np.zeros((480, 640))
    for mask in masks:
        fg_mask += mask
    fg_mask = fg_mask > 0
    return fg_mask.astype(np.uint8)


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = 480, 640
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def masks_to_boundary(masks, dilation_ratio=0.01):
    # masks = [N, H, W], numpy array
    fg_mask = masks_to_fg_mask(masks)
    boundary = np.zeros((480, 640))
    for mask in masks:
        boundary += mask_to_boundary(mask, dilation_ratio=dilation_ratio)
    boundary = boundary > 0
    return boundary.astype(np.uint8)

def mask_to_rle(mask):
    rle = m.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

# whether from the PNG are used or new colors are generated
generate_new_colors = True

json_file = 'detectron2_datasets/TODv2/annotations/tod_v2_train_panoptic_perturbated.json'
segmentations_folder = './detectron2_datasets/TODv2/'
img_folder = './detectron2_datasets/TODv2/'

with open(json_file, 'r') as f:
    coco_anno = json.load(f)


# get image info
img_infos = coco_anno['images']
annos = coco_anno['annotations']

new_img_infos = []
for img_info, anno in zip(img_infos, tqdm(annos)):
    segments_info = anno['segments_info']
    pan_seg_gt = utils.read_image(os.path.join(segmentations_folder, img_info["file_name"].replace(".jpeg", '.png')), "RGB")
    panoptic = rgb2id(pan_seg_gt)
    gt_masks = []
    for seg in segments_info:
        gt_mask = panoptic == seg["id"]
        gt_masks.append(gt_mask.astype(np.uint8)*255)
    gt_masks = np.array(gt_masks)
    perturbed_segms = img_info["perturbed_segmentation"]
    perturbed_masks = []
    for segm in perturbed_segms:
        if isinstance(segm, dict):
            perturbed_masks.append(mask_util.decode(segm))
        else:
            raise ValueError(
                "Cannot convert segmentation of type '{}' to BitMasks!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict, or a binary segmentation mask "
                " in a 2D numpy array of shape HxW.".format(type(segm))
            )
    
       
    gt_fg_mask = masks_to_fg_mask(gt_masks)
    gt_boundary = masks_to_boundary(gt_masks)
    perturbed_fg_mask = masks_to_fg_mask(perturbed_masks)
    perturbed_boundary = masks_to_boundary(perturbed_masks)

    tp_mask = np.logical_and(gt_fg_mask, perturbed_fg_mask)
    tn_mask = np.logical_and(np.logical_not(gt_fg_mask), np.logical_not(perturbed_fg_mask))
    fp_mask = np.logical_and(np.logical_not(gt_fg_mask), perturbed_fg_mask)
    fn_mask = np.logical_and(gt_fg_mask, np.logical_not(perturbed_fg_mask))

    # print(np.unique(gt_fg_mask))
    # cv2.imwrite('test.png', np.vstack([gt_fg_mask*255, perturbed_fg_mask*255, tp_mask*255, tn_mask*255, fp_mask*255, fn_mask*255]))
    # exit()

    tp_boundary = np.logical_and(gt_boundary, perturbed_boundary)
    tn_boundary = np.logical_and(np.logical_not(gt_boundary), np.logical_not(perturbed_boundary))
    fp_boundary = np.logical_and(np.logical_not(gt_boundary), perturbed_boundary)
    fn_boundary = np.logical_and(gt_boundary, np.logical_not(perturbed_boundary))
    
    img_info["tp_mask"] = mask_to_rle(np.array(tp_mask, dtype=bool, order='F'))
    img_info["tn_mask"] = mask_to_rle(np.array(tn_mask, dtype=bool, order='F'))
    img_info["fp_mask"] = mask_to_rle(np.array(fp_mask, dtype=bool, order='F'))
    img_info["fn_mask"] = mask_to_rle(np.array(fn_mask, dtype=bool, order='F'))
    img_info["tp_boundary"] = mask_to_rle(np.array(tp_boundary, dtype=bool, order='F'))
    img_info["tn_boundary"] = mask_to_rle(np.array(tn_boundary, dtype=bool, order='F'))
    img_info["fp_boundary"] = mask_to_rle(np.array(fp_boundary, dtype=bool, order='F'))
    img_info["fn_boundary"] = mask_to_rle(np.array(fn_boundary, dtype=bool, order='F'))
    new_img_infos.append(img_info)

coco_anno['images'] = new_img_infos

with open(json_file, 'w') as f:
    json.dump(coco_anno, f)