import os
import numpy as np
import torch
from pycocotools.coco import COCO
import cv2


## combine different datasets into one
root = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote'
ann_json = os.path.join(root, 'train.json')

coco = COCO(ann_json)
# ids = list(coco.anns.keys())  # list of all annotation ids in the dataset
ids = coco.getAnnIds(catIds=[2])
print(len(ids))
img_ids = list(coco.imgs.keys())


coco = coco
ann_id = ids[0]
img_id = coco.anns[ann_id]['image_id']
path = coco.loadImgs(img_id)[0]['file_name']
im = cv2.imread(os.path.join(root, 'images', path))

ann = coco.anns[ann_id]
gt = coco.annToMask(ann)
cv2.imwrite('im.png', im)
cv2.imwrite('gt.png', gt.astype(np.uint8) * 255)
        