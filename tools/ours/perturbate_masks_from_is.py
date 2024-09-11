import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
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

uoais_sim_path = '/home/work/Workspace/seung/mask-refiner/detectron2_datasets/UOAIS-Sim'
results_path = '/home/work/Workspace/seung/mask-refiner/results'
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

purturbed_coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'coco_anns_uoais_sim_{}_perturbed_is.json'.format(split))
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

    
    def mask_to_rle(mask):
        rle = m.encode(mask)
        rle['counts'] = rle['counts'].decode('ascii')
        return rle
    if img_id < 11250:
        is_model = 'ucn'
    elif img_id < 22500:
        is_model = 'uoaisnet'
    elif img_id < 33750:
        is_model = 'msmformer'
    else: 
        is_model = 'uoisnet3d'
    is_path = os.path.join(results_path, is_model, 'npy', '{}.npy'.format(img_id))
    perturbated_masks = np.load(is_path)

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
