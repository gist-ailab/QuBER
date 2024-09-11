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
import imageio
import json

uoais_sim_path = '/SSDa/workspace/seunghyeok_back/mask-refiner/detectron2_datasets/UOAIS-Sim'
split = 'val'
n_random_instance = [3, 10]
coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'coco_anns_uoais_sim_{}.json'.format(split))
coco_anno = COCO(coco_anno_path)
img_ids = coco_anno.getImgIds()

augmented_coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'coco_anns_uoais_sim_{}_augmented.json'.format(split))
augmented_coco_anno_json = {
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

def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) ==0:
        return None, None, None, None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)

def load_random_instances(coco_anno):

    img_ids = coco_anno.getImgIds()
    while True:
        # select random img_id 
        img_id = random.choice(img_ids)
        img_info = coco_anno.loadImgs(img_id)[0]
        anno_ids = coco_anno.getAnnIds(imgIds=img_id)
        annos = coco_anno.loadAnns(anno_ids)
        random.shuffle(annos)
        for anno in annos:
            if anno['occluded_rate'] < 0.05:
                rgb = cv2.imread(os.path.join(uoais_sim_path, split, img_info['file_name']))
                depth = imageio.imread(os.path.join(uoais_sim_path, split, img_info['depth_file_name']))
                visible_mask = m.decode(anno['visible_mask'])
                x, y, w, h = anno['visible_bbox']
                rgb = rgb * visible_mask[:, :, None]
                depth = depth * visible_mask
                rgb_crop = rgb[y:y+h, x:x+w]
                depth_crop = depth[y:y+h, x:x+w]
                visible_crop = visible_mask[y:y+h, x:x+w]
                return rgb_crop, depth_crop, visible_crop




new_anno_id = 1
new_img_id = 1
augmented_annos = []
augmented_image_infos = []
for img_id in tqdm(img_ids):

    for _ in range(4):

        # load COCO GT
        anno_ids = coco_anno.getAnnIds(imgIds=img_id)
        annos = coco_anno.loadAnns(anno_ids)
        img_info = coco_anno.loadImgs(img_id)[0]
        rgb_path = os.path.join(uoais_sim_path, split, img_info['file_name'])
        rgb = cv2.imread(rgb_path)
        depth_path = os.path.join(uoais_sim_path, split, img_info['depth_file_name'])
        depth = imageio.imread(depth_path)

        n_added_instance = random.randint(n_random_instance[0], n_random_instance[1])
        added_instances = []
        for _ in range(n_added_instance):
            rgb_crop, depth_crop, mask_crop = load_random_instances(coco_anno)
            added_instances.append((rgb_crop, depth_crop, mask_crop))


        gt_masks = [m.decode(anno['visible_mask']) for anno in annos]
        # gt_masks: [num_instances, height, width], values in {0, 1}
        # visualizer = Visualizer(img)
        # viz = visualizer.overlay_instances(masks=gt_masks, alpha=1.0)
        # ori_gt_viz = viz.get_image()

        # select random gt instance, and add random instance
        blended_rgb = rgb.copy()
        blended_depth = depth.copy()
        for idx in range(n_added_instance):
            for _ in range(10):
                random_gt_idx = random.randint(0, len(gt_masks)-1)
                gt_mask = gt_masks[random_gt_idx]
                x_gt, y_gt, w_gt, h_gt = get_bbox(gt_mask)
                if x_gt is None:
                    continue

                # select random instance
                rgb_crop, depth_crop, mask_crop = added_instances[idx]
                h, w = mask_crop.shape
                x_to_add = random.randint(max(int(x_gt - w_gt *0.5), 0), min(int(x_gt + w_gt*0.5), rgb.shape[1]-1))
                y_to_add = random.randint(max(int(y_gt - h_gt *0.5), 0), min(int(y_gt + h_gt*0.5), rgb.shape[0]-1))

                # resize the width and height of the random instance
                z = depth[y_to_add, x_to_add]
                z_median = np.median(depth_crop[mask_crop > 0])
                w = int(w * z_median / z)
                h = int(h * z_median / z)
                if z == 0 or z_median == 0 or w == 0 or h == 0:
                    continue
                rgb_crop = cv2.resize(rgb_crop, (w, h))
                mask_crop = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
                depth_crop = cv2.resize(depth_crop, (w, h), interpolation=cv2.INTER_NEAREST)

                if random.random() < 0.5:
                    rgb_crop = cv2.GaussianBlur(rgb_crop, (5, 5), 2)
                    # mask_crop = np.where(rgb_crop > 0, 1, 0)[:, :, 0]

                rgb_to_add = np.zeros_like(rgb)
                mask_to_add = np.zeros_like(gt_mask)
                depth_to_add = np.zeros_like(depth)

                if x_to_add + w > rgb.shape[1]:
                    w = rgb.shape[1] - x_to_add
                    rgb_crop = rgb_crop[:, :w]
                    mask_crop = mask_crop[:, :w]
                    depth_crop = depth_crop[:, :w]
                if y_to_add + h > rgb.shape[0]:
                    h = rgb.shape[0] - y_to_add
                    rgb_crop = rgb_crop[:h]
                    mask_crop = mask_crop[:h]
                    depth_crop = depth_crop[:h]
                rgb_to_add[y_to_add:y_to_add+h, x_to_add:x_to_add+w] = rgb_crop
                mask_to_add[y_to_add:y_to_add+h, x_to_add:x_to_add+w] = mask_crop
                depth_to_add[y_to_add:y_to_add+h, x_to_add:x_to_add+w] = depth_crop + z - z_median
                overlaps = []
                for mask in gt_masks:
                    overlap = np.logical_and(mask_to_add, mask)
                    overlaps.append(overlap)
                overlap = np.logical_or.reduce(overlaps)
                if np.sum(overlap) < 50:
                    continue
                blended_rgb = np.where(mask_to_add[:, :, None], rgb_to_add, blended_rgb)   
                blended_depth = np.where(mask_to_add, depth_to_add, blended_depth)

                for k in range(len(gt_masks)):
                    gt_masks[k] = np.logical_and(gt_masks[k], ~overlap)
                # gt_masks[random_gt_idx] = gt_mask
                gt_masks.append(mask_to_add)
                break




        def mask_to_rle(mask):
            rle = m.encode(mask)
            rle['counts'] = rle['counts'].decode('ascii')
            return rle
        
        # convert to coco format
        for gt_mask in gt_masks:
            augmented_anno = {
                'id': int(new_anno_id),
                'image_id': int(new_img_id),
                'category_id': 1,
                'segmentation': mask_to_rle(np.array(gt_mask, dtype=bool, order='f')),
                'bbox': get_bbox(gt_mask),
                'area': int(gt_mask.sum()),
                'iscrowd': 0,
                'height': 480,
                'width': 640,
            }
            new_anno_id += 1
            augmented_annos.append(augmented_anno)
        augmented_image_info = img_info.copy()
        augmented_image_info['id'] = int(new_img_id)
        file_name = os.path.basename(img_info['file_name'])
        augmented_image_info['file_name'] = augmented_image_info['file_name'].replace('bin', 'bin_aug').replace('tabletop', 'tabletop_aug').replace(file_name, '{}.png'.format(new_img_id))
        augmented_image_info['depth_file_name'] = augmented_image_info['depth_file_name'].replace('bin', 'bin_aug').replace('tabletop', 'tabletop_aug').replace(file_name, '{}.png'.format(new_img_id))

        new_rgb_path = os.path.join(uoais_sim_path, split, augmented_image_info['file_name'])
        new_depth_path = os.path.join(uoais_sim_path, split, augmented_image_info['depth_file_name'])
        cv2.imwrite(new_rgb_path, blended_rgb)
        cv2.imwrite(new_depth_path, blended_depth)

        augmented_image_infos.append(augmented_image_info)
        new_img_id = new_img_id + 1


augmented_coco_anno_json['images'] = augmented_image_infos
augmented_coco_anno_json['annotations'] = augmented_annos
with open(augmented_coco_anno_path, 'w') as f:
    json.dump(augmented_coco_anno_json, f, indent=4)
