
import argparse
import numpy as np
import os
import cv2

from eval_utils import run_eval
from pycocotools.coco import COCO
from eval.refiner_model import *
import imgviz

# if __name__ == "__main__":

armbench_root = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench'
coco_json_path = os.path.join(armbench_root, 'mix-object-tote', 'test.json')


coco = COCO(coco_json_path)
imgIds = coco.getImgIds(catIds=[2])


config_file = '/SSDe/seunghyeok_back/mask-refiner/configs/armbench/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml'
weights_file = 'model_final.pth'
refiner_moder = MaskRefiner(config_file, weights_file, dataset='armbench')

for imgid in imgIds:
    img_info = coco.loadImgs(imgid)[0]
    if img_info['file_name'] != 'N210NHGDKS.jpg':
        continue
    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[2])
    anns = coco.loadAnns(annIds)

    img_path = os.path.join(armbench_root, 'mix-object-tote', 'images', img_info['file_name'])
    img = cv2.imread(img_path)

    initial_masks = np.load('vis_npy/{}.npy'.format(img_info['file_name'].split('.')[0])) # (N, H, W)
    H, W = initial_masks.shape[1], initial_masks.shape[2]
    refined_masks, refined_output, refined_pred_time, fg_mask = refiner_moder.predict(img_path, None, initial_masks, None)
    initial_vis = imgviz.instances2rgb(img.copy(), masks=initial_masks, labels=list(range(initial_masks.shape[0])), line_width=0, boundary_width=3)
    img = cv2.resize(img, (refined_masks.shape[2], refined_masks.shape[1]))
    refine_vis = imgviz.instances2rgb(img.copy(), masks=refined_masks, labels=list(range(refined_masks.shape[0])), line_width=0, boundary_width=3)
    cv2.imwrite('1.jpg', initial_vis)
    cv2.imwrite('2.jpg', refine_vis)


    


    
    break


