import glob
import cv2
import numpy as np

import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
from pycocotools import mask as m
import datetime
from tqdm import tqdm
import imageio



base_dir = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/training_set'
coco_json_path = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/annotations/tod_v2_train.json'


coco_json = {
    "info": {
        "description": "TOD v2",
        "url": "https://github.com/chrisdxie/rice",
        "version": "0.1.0",
        "year": 2023,
        "contributor": "chris xie et al. this coco file is generated by seunghyeok back",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial 4.0 International",
            "url": ""
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

annotations = []
image_infos = []
annotation_id = 1
img_id = 1

def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) ==0:
        return None, None, None, None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)

def mask_to_rle(mask):
    rle = m.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

scene_dirs = sorted(glob.glob(base_dir + '/*'))
for scene_dir in tqdm(scene_dirs):
    for view_num in range(2, 7):
        rgb_img_filename = scene_dir + f"/rgb_{view_num:05d}.jpeg"
        rgb_path = rgb_img_filename.replace(base_dir, 'training_set')
        
        depth_img_filename = scene_dir + f"/depth_{view_num:05d}.png"
        depth_path = depth_img_filename.replace(base_dir, 'training_set')
        
        foreground_mask_filename = scene_dir + f"/segmentation_{view_num:05d}.png"
        foreground_mask_path = foreground_mask_filename.replace(base_dir, 'training_set')
        foreground_mask = imageio.imread(foreground_mask_filename) 
        colors = np.unique(foreground_mask.reshape(-1, 3), axis=0)
        exclude_colors = set([(0, 0, 0), (128, 0, 0)])
        colors = [color for color in colors if tuple(color) not in exclude_colors]
        h, w = foreground_mask.shape[:2]
        for color in colors:            
            mask = np.equal(foreground_mask, color).all(axis=-1)
            mask = np.array(mask, dtype=bool, order='F')
            bbox = get_bbox(mask)
            if bbox[0] is None:
                continue
            
            annotation = {}
            annotation["id"] = annotation_id
            annotation["image_id"] = img_id
            annotation["category_id"] = 1
            annotation["bbox"] = bbox
            annotation["height"] = int(h)
            annotation["width"] = int(w)
            annotation["iscrowd"] = 0
            annotation["segmentation"] = mask_to_rle(mask)
            annotation["area"] = int(np.sum(mask))
            annotation_id += 1
            annotations.append(annotation)
            image_infos.append(
                {
                    "id": img_id,
                    "file_name": rgb_path,
                    "depth_file_name": depth_path,
                    "width": int(w),
                    "height": int(h),
                    "date_capture": "None",
                    "license": 1,
                    "coco_url": "None",
                    "flickr_url": "None",
                }
            )
        img_id += 1
coco_json["annotations"] = annotations
coco_json["images"] = image_infos
with open(coco_json_path, "w") as f:
    print("Saving annotation as COCO format to", coco_json_path)
    json.dump(coco_json, f)