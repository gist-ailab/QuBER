import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation import segment
import cv2
import random
import time
import os
from pycocotools.coco import COCO
from pycocotools import mask as m

from detectron2.utils.visualizer import Visualizer
from perturbation_utils import *
from tqdm import tqdm
import datetime

import json

uoais_sim_path = '/SSDc/Workspaces/seunghyeok_back/mask-refiner/UOAIS-Sim'
split = 'train'

# parameters
fp_ratio_range = [0.0, 0.2]
gs_ratio_range = [0.0, 0.3]
merge_ratio_range = [0.0, 0.1]
delte_ratio_range = [0.0, 0.1]
split_ratio_range = [0.0, 0.1]
iou_target_range = [0.8, 1.0]
min_mask_ratio = 0.01
coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'coco_anns_uoais_sim_{}.json'.format(split))
coco_anno = COCO(coco_anno_path)
img_ids = coco_anno.getImgIds()

purturbed_coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'coco_anns_uoais_sim_{}_perturbed.json'.format(split))
purturbed_coco_anno_json = {
    "info": {
        "description": "UOAIS-SIM Mask Refiner Dataset",
        "url": "https://github.com/gist-ailab/uoais",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "Seunghyeok Back",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "categories": [
        {
            'id': 1,
            'name': 'object',
            'supercategory': 'shape',
        }
    ],
    "images": [],
    "annotations": []
}


purturbed_image_infos = []
perturbed_annos = []

for img_id in tqdm(img_ids):

    fp_ratio = random.uniform(*fp_ratio_range)
    gs_ratio = random.uniform(*gs_ratio_range)
    merge_ratio = random.uniform(*merge_ratio_range)
    delete_ratio = random.uniform(*delte_ratio_range)
    split_ratio = random.uniform(*split_ratio_range)

    # load COCO GT
    anno_ids = coco_anno.getAnnIds(imgIds=img_id)
    annos = coco_anno.loadAnns(anno_ids)
    img_info = coco_anno.loadImgs(img_id)[0]
    img_path = os.path.join(uoais_sim_path, split, img_info['file_name'])
    img = cv2.imread(img_path)

    # gt_masks: [num_instances, height, width], values in {0, 1}
    gt_masks = np.array([m.decode(anno['visible_mask']) for anno in annos])
    # visualizer = Visualizer(img)
    # viz = visualizer.overlay_instances(masks=gt_masks, alpha=1.0)
    # ori_gt_viz = viz.get_image()

    # apply efficient graph-based segmentation,  values in {0, 1}
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (w//4, h//4))
    gs_masks = segment(np.array(img_resized), 0.2, 50, 50) # [height, width] with values in [0, 255]
    gs_masks = cv2.resize(gs_masks, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # gs_masks: [num_instances, height, width]
    gs_masks = np.array([gs_masks == i for i in np.unique(gs_masks)[1:]], dtype=np.uint8)[:, :, :, 0]
    # visualizer = Visualizer(img)
    # viz = visualizer.overlay_instances(masks=gs_masks, alpha=1.0)
    # gs_viz = viz.get_image()


    perturbated_masks = []
    ## Add false positives
    # select n_fp gs_masks that is not overlapped with gt_masks

    max_gt_mask_area = np.max([np.sum(gt_mask) for gt_mask in gt_masks])
    for idx in np.arange(len(gs_masks)):
        gs_mask = gs_masks[idx]
        if random.random() > fp_ratio:
            continue
        # filter out too small and large masks
        if np.sum(gs_mask) < (w * h * min_mask_ratio) or np.sum(gs_mask) > max_gt_mask_area * 2.0:
            continue
        max_iou = 0
        for gt_mask in gt_masks:
            iou = compute_iou(gt_mask, gs_mask)
            max_iou = max(max_iou, iou)
        if max_iou < 0.3:
            perturbated_masks.append(gs_mask*255)

    ## Add under / over-segmentation 
    for idx in np.arange(len(gs_masks)):
        if random.random() > gs_ratio:
            continue
        gs_mask = gs_masks[idx]
        # filter out toosmall masks
        if np.sum(gs_mask) < (w * h * min_mask_ratio):
            continue
        max_iou = 0
        for idx, gt_mask in enumerate(gt_masks):
            iou = compute_iou(gt_mask, gs_mask)
            max_iou = max(max_iou, iou)
        if max_iou > 0.3:
            perturbated_masks.append(gs_mask*255)
    
    # select not used gt_masks, and append to perturbated_masks
    _gt_masks = []
    for gt_mask in gt_masks:
        max_iou = 0
        for perturbated_mask in perturbated_masks:
            iou = compute_iou(gt_mask, perturbated_mask)
            max_iou = max(max_iou, iou)
        if max_iou < 0.3:
            _gt_masks.append(gt_mask*255)
    perturbated_masks.extend(_gt_masks)

    # merge close masks that is close to each other within 10 pixels
    for idx1 in np.arange(len(perturbated_masks)):
        if random.random() > merge_ratio:
            continue
        mask1 = perturbated_masks[idx1]
        for idx2 in np.arange(len(perturbated_masks)):
            if idx1 == idx2:
                continue
            mask2 = perturbated_masks[idx2]
            # check whether two masks are close to each other via dilation
            dilated_mask = cv2.dilate(mask1.copy(), np.ones((10, 10), np.uint8), iterations=1)
            if np.sum(dilated_mask * mask2) > 0:
                perturbated_masks[idx1] = mask1 + mask2
                perturbated_masks[idx2] = np.zeros_like(mask2)
    # remove empty masks
    perturbated_masks = [mask for mask in perturbated_masks if np.sum(mask) > 0]
    
    # split masks by sampling a random line
    for idx in np.arange(len(perturbated_masks)):
        if random.random() > split_ratio:
            continue
        valid = False
        for k in range(10):
            mask = perturbated_masks[idx]
            # find a random line
            y, x = np.where(mask != 1)
            x_min, y_min = np.min(x), np.min(y)
            x_max, y_max = np.max(x), np.max(y)
            x1, y1 = random.randint(x_min, x_max), random.randint(y_min, y_max)
            x2, y2 = random.randint(x_min, x_max), random.randint(y_min, y_max)
            # split the mask into two parts
            mask1 = mask.copy()
            if random.random() < 0.5:
                if random.random() < 0.5:
                    mask1[y1:y_max, :] = 0
                else:
                    mask1[y_min:y1, :] = 0
            else:
                if random.random() < 0.5:
                    mask1[:, x1:x_max] = 0
                else:
                    mask1[:, x_min:x1] = 0
            mask2 = mask.copy()
            mask2 = np.where(mask1 !=0, 0, mask2)
            if np.sum(mask1) < (w * h * min_mask_ratio) or np.sum(mask2) < (w * h * min_mask_ratio):
                continue
            valid = True
            break
        if valid:
            perturbated_masks[idx] = mask1
            perturbated_masks.append(mask2)


    # delete masks
    del_indices = []
    for idx in np.arange(len(perturbated_masks)):
        if random.random() > delete_ratio:
            continue
        del_indices.append(idx)
    perturbated_masks = [mask for idx, mask in enumerate(perturbated_masks) if idx not in del_indices]

    # add random purturbation
    for idx in range(len(perturbated_masks)):
        iou_target = random.uniform(*iou_target_range)
        perturbated_masks[idx] = modify_boundary(perturbated_masks[idx], iou_target=iou_target)

    perturbated_masks = np.array(perturbated_masks)

    # visualizer = Visualizer(img)
    # viz = visualizer.overlay_instances(masks=perturbated_masks, alpha=1.0)
    # fp_viz = viz.get_image()

    # cv2.imwrite('img.png', np.hstack([img, ori_gt_viz, fp_viz]))
    
    def mask_to_rle(mask):
        rle = m.encode(mask)
        rle['counts'] = rle['counts'].decode('ascii')
        return rle
    
    # convert to coco format
    for anno in annos:
        perturbed_anno = {
            'id': anno['id'],
            'image_id': anno['image_id'],
            'category_id': anno['category_id'],
            'segmentation': anno['visible_mask'],
            'bbox': anno['bbox'],
            'area': int(np.sum(m.decode(anno['visible_mask']))),
            'iscrowd': 0,
            'height': anno['height'],
            'width': anno['width'],
        }
        perturbed_annos.append(perturbed_anno)
    purturbed_image_info = img_info.copy()
    purturbed_image_info['perturbed_segmentation'] = [mask_to_rle(np.array(mask, dtype=bool, order='f')) for mask in perturbated_masks]
    purturbed_image_infos.append(purturbed_image_info)

purturbed_coco_anno_json['images'] = purturbed_image_infos
purturbed_coco_anno_json['annotations'] = perturbed_annos
with open(purturbed_coco_anno_path, 'w') as f:
    json.dump(purturbed_coco_anno_json, f, indent=4)
