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


base_dir = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2'
perturbed_coco_anno_path = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/annotations/tod_v2_train_perturbated.json'
perturbed_coco_anno = COCO(perturbed_coco_anno_path)
img_ids = perturbed_coco_anno.getImgIds()

for i in range(10):
    img_id = random.choice(img_ids)

    # 

    img_info = perturbed_coco_anno.loadImgs(img_id)[0]
    img_path = os.path.join(base_dir, img_info['file_name'])

    img = cv2.imread(img_path)

    # visualize perturbed masks
    perturbed_masks = img_info['perturbed_segmentation']
    perturbed_masks = [m.decode(perturbed_mask) for perturbed_mask in perturbed_masks]
    visualizer = Visualizer(img)
    viz = visualizer.overlay_instances(masks=perturbed_masks, alpha=1.0)
    perturbed_mask_viz = viz.get_image()

    # visualize gt masks
    anno_ids = perturbed_coco_anno.getAnnIds(imgIds=img_id)
    annos = perturbed_coco_anno.loadAnns(anno_ids)
    gt_masks = np.array([perturbed_coco_anno.annToMask(anno) for anno in annos])
    visualizer= Visualizer(img)
    viz = visualizer.overlay_instances(masks=gt_masks, alpha= 1.0)
    gt_mask_viz = viz.get_image()

    


    cv2.imwrite('perturbed_mask_viz_{}.png'.format(i), np.hstack([img, perturbed_mask_viz, gt_mask_viz, tp_mask_viz]))