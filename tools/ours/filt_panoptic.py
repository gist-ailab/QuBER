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


json_file = 'detectron2_datasets/UOAIS-Sim/annotations/uoais_sim_train_panoptic_augmented_perturbed.json'
segmentations_folder = './detectron2_datasets/UOAIS-Sim/train/'
img_folder = './detectron2_datasets/UOAIS-Sim/train/'

with open(json_file, 'r') as f:
    coco_anno = json.load(f)


# get image info
img_infos = coco_anno['images']
annos = coco_anno['annotations']

new_img_infos = []
new_annos = []
# get only first 5000 images and remove the rest
print(len(img_infos), len(annos))
for img_info, anno in zip(img_infos, tqdm(annos)):
    img_id = img_info['id']
    if img_id > 45000:
        continue
    else:
        new_img_infos.append(img_info)
        new_annos.append(anno)


coco_anno['images'] = new_img_infos
coco_anno['annotations'] = new_annos

with open(json_file, 'w') as f:
    json.dump(coco_anno, f)