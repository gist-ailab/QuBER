import argparse
import os
import sys
import cv2
import glob
import numpy as np
import imageio
import torch
import yaml
from util import *
from tqdm import tqdm
from adet.utils.post_process import detector_postprocess, DefaultPredictor
import torch.nn as nn
from torchvision.transforms import Normalize

from models.late_fusion import create_late_fusion_model

BACKGROUND_LABEL = 0

if __name__ == "__main__":

    uoais_config_path = '/SSDe/seunghyeok_back/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml'
    cgnet_weight_path = '/SSDe/seunghyeok_back/uoais/foreground_segmentation/rgbd_fg.pth'
    osd_path = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/OSD-0.2-depth'
    gpu_id = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # load uoais and foreground segmentation model
    init_segm_model, det_cfg = load_uoais(uoais_config_path)
    fg_model = load_cgnet(cgnet_weight_path)
    W, H = det_cfg.INPUT.IMG_SIZE

    # load dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(osd_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(osd_path)))
    anno_paths = sorted(glob.glob("{}/annotation/*.png".format(osd_path)))
    assert len(rgb_paths) == len(depth_paths)
    assert len(rgb_paths) == len(anno_paths)
    print("Evaluation on OSD dataset: {} rgbs, {} depths, {} visible masks".format(
                len(rgb_paths), len(depth_paths), len(anno_paths)), "green")
    

    for idx, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, anno_paths)):

        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if det_cfg.INPUT.DEPTH and det_cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif det_cfg.INPUT.DEPTH and not det_cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        else:
            uoais_input = rgb_img
        # laod GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]

        # forward (UOAIS)
        outputs = init_segm_model(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        if det_cfg.INPUT.AMODAL:
            pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        else:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            
        # CG-Net inference
        fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = cv2.resize(depth_img, (320, 240)) 
        fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)
    
        # get initial masks
        initial_masks = np.zeros([H, W])  # [N, H, W]
        for i, mask in enumerate(pred_masks):
            iou = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
            if iou >= 0.5:
                initial_masks[mask] = i+1
        output_file_name = rgb_path.split('/')[-1]
        output_file_path = os.path.join(osd_path, 'uoais', output_file_name)
        imageio.imwrite(output_file_path, initial_masks.astype(np.uint8))
