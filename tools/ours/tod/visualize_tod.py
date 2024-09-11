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
from tqdm import tqdm
import imageio

uoais_sim_path = '/SSDa/workspace/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2'
coco_json_path = '/SSDa/workspace/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/annotations/tod_v2_train.json'
augmented_coco_anno = COCO(coco_json_path)
img_ids = augmented_coco_anno.getImgIds()

for i in range(10):
    img_id = random.choice(img_ids)

    img_info = augmented_coco_anno.loadImgs(img_id)[0]
    img_path = os.path.join(uoais_sim_path, img_info['file_name'])
    img = cv2.imread(img_path)

    depth_path = os.path.join(uoais_sim_path, img_info['depth_file_name'])
    depth = imageio.imread(depth_path)
    print(depth.min(), depth.max())
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    depth = np.stack([depth, depth, depth], axis=-1)
    # visualize gt masks
    anno_ids = augmented_coco_anno.getAnnIds(imgIds=img_id)
    annos = augmented_coco_anno.loadAnns(anno_ids)
    gt_masks = np.array([augmented_coco_anno.annToMask(anno) for anno in annos])
    visualizer= Visualizer(img)
    viz = visualizer.overlay_instances(masks=gt_masks, alpha=1.0)
    gt_mask_viz = viz.get_image()

    cv2.imwrite('augmented_{}.png'.format(img_id), np.vstack([img, depth, gt_mask_viz]))