#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data
The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import imageio
from panopticapi.utils import IdGenerator, rgb2id
from pycocotools import mask as m
from detectron2.utils.visualizer import Visualizer

# whether from the PNG are used or new colors are generated
generate_new_colors = True

json_file = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/annotations/tod_v2_train_panoptic_perturbated.json'
segmentations_folder = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/'
img_folder = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/TODv2/'
panoptic_coco_categories = 'detectron2_datasets/UOAIS-Sim/annotations/panoptic_uoais_sim_categories.json'

with open(json_file, 'r') as f:
    coco_d = json.load(f)

ann = np.random.choice(coco_d['annotations'])

with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}

# find input img that correspond to the annotation
img = None
for image_info in coco_d['images']:
    if image_info['id'] == ann['image_id']:
        try:
            img = np.array(
                Image.open(os.path.join(img_folder, image_info['file_name']))
            )
            depth_img = imageio.imread(os.path.join(img_folder, image_info['depth_file_name']))
            depth_img[depth_img < 500] = 500
            depth_img[depth_img > 1500] = 1500
            depth_img = (depth_img - 500) / (1500 - 500)
            depth_img = np.uint8(depth_img * 255)

            tp_mask = m.decode(image_info['tp_mask'])
            tn_mask = m.decode(image_info['tn_mask'])
            fp_mask = m.decode(image_info['fp_mask'])
            fn_mask = m.decode(image_info['fn_mask'])
            perturbed_masks = image_info['perturbed_segmentation']
            perturbed_masks = [m.decode(perturbed_mask) for perturbed_mask in perturbed_masks]
            visualizer = Visualizer(img)
            viz = visualizer.overlay_instances(masks=perturbed_masks, alpha=1.0)
            perturbed_mask_viz = viz.get_image()

        except:
            print("Undable to find correspoding input image.")
        break

segmentation = np.array(
    Image.open(os.path.join(segmentations_folder, ann['file_name'])),
    dtype=np.uint8
)
segmentation_id = rgb2id(segmentation)
# find segments boundaries
boundaries = find_boundaries(segmentation_id, mode='thick')

if generate_new_colors:
    segmentation[:, :, :] = 0
    color_generator = IdGenerator(categegories)
    for segment_info in ann['segments_info']:
        color = color_generator.get_color(segment_info['category_id'])
        mask = segmentation_id == segment_info['id']
        segmentation[mask] = color

# depict boundaries
segmentation[boundaries] = [0, 0, 0]


if img is None:
    plt.figure()
    plt.imshow(segmentation)
    plt.axis('off')
else:
    plt.figure(figsize=(9, 5))
    plt.subplot(181)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(182)
    plt.imshow(segmentation)
    plt.axis('off')
    plt.subplot(183)
    plt.imshow(depth_img)
    plt.axis('off')
    plt.subplot(184)
    plt.imshow(tp_mask)
    plt.axis('off')
    plt.subplot(185)
    plt.imshow(tn_mask)
    plt.axis('off')
    plt.subplot(186)
    plt.imshow(fp_mask)
    plt.axis('off')
    plt.subplot(187)
    plt.imshow(fn_mask)
    plt.axis('off')
    plt.subplot(188)
    plt.imshow(perturbed_mask_viz)
    plt.axis('off')
    plt.tight_layout()

plt.savefig('panoptic_example.png')