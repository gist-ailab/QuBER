# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
import torch.nn.functional as F
from panopticapi.utils import rgb2id
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, polygons_to_bitmask
from detectron2.projects.point_rend import ColorAugSSDTransform

from .target_generator import PanopticDeepLabTargetGenerator, PerturbedInputGenerator
from .augmentation import PerlinDistortion

__all__ = ["PerturbedPanopticDatasetMapper"]

import imageio
import cv2

class PerturbedPanopticDatasetMapper:
    """
    The callable currently does the following:
    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        panoptic_target_generator: Callable,
        perturbed_input_generator: Callable,
        offset_input_on: bool,
        eee_mask_on: bool,
        eee_boundary_on: bool,
        depth_on: bool,
        depth_range: List[float],
        perlin_distortion_on: bool,
        rgb_on: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Transform used in {}: {}".format("train" if is_train else "eval", str(augmentations)))

        self.panoptic_target_generator = panoptic_target_generator
        self.perturbed_input_generator = perturbed_input_generator
        
        self.offset_input_on = offset_input_on
        self.eee_mask_on = eee_mask_on
        self.eee_boundary_on = eee_boundary_on

        self.depth_on = depth_on
        self.depth_min = depth_range[0]
        self.depth_max = depth_range[1]
        self.perlin_distortion_on = perlin_distortion_on
        self.rgb_on = rgb_on


    @classmethod
    def from_config(cls, cfg, is_train):

        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.COLOR_AUG_SSD and is_train and not (cfg.INPUT.DEPTH_ON and not cfg.INPUT.RGB_ON):
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        # !TODO: support crop and flip augmentation
        # if cfg.INPUT.CROP.ENABLED:
        #     augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        # augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )
        perturbed_input_generator = PerturbedInputGenerator(
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
        )
        

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "perturbed_input_generator": perturbed_input_generator,
            "eee_mask_on":  cfg.MODEL.INS_EMBED_HEAD.EEE_MASK_ON,
            "eee_boundary_on": cfg.MODEL.INS_EMBED_HEAD.EEE_BOUNDARY_ON,
            "offset_input_on": cfg.INPUT.OFFSET_INPUT_ON,
            "depth_on": cfg.INPUT.DEPTH_ON,
            "depth_range": cfg.INPUT.DEPTH_RANGE,
            "perlin_distortion_on": cfg.INPUT.PERLIN_DISTORTION_ON,
            "rgb_on": cfg.INPUT.RGB_ON
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        if self.rgb_on:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            ori_h, ori_w = image.shape[:2]
            utils.check_image_size(dataset_dict, image)
            dataset_dict["height"] = image.shape[0]
            dataset_dict["width"] = image.shape[1]

        if self.depth_on:
            depth = imageio.imread(dataset_dict["depth_file_name"]).astype(np.float32)
            utils.check_image_size(dataset_dict, depth)
            if self.perlin_distortion_on and self.is_train:
                depth = PerlinDistortion(depth)
            depth[depth > self.depth_max] = self.depth_max
            depth[depth < self.depth_min] = self.depth_min
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min) * 255
            depth = np.expand_dims(depth, -1)
            depth = np.uint8(np.repeat(depth, 3, -1))

        # Panoptic label is encoded in RGB image.
        try:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        except:
            pans_seg_gt_path = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/test/{}.png'.format(dataset_dict["file_name"].split('/')[-1].split('.')[0])  
            pan_seg_gt = utils.read_image(pans_seg_gt_path, "RGB")
        # Reuses semantic transform for panoptic labels.

        if self.rgb_on and self.depth_on:
            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            aug_input, transformation = T.apply_transform_gens(self.augmentations, aug_input)
            image, pan_seg_gt = aug_input.image, aug_input.sem_seg
            depth = transformation.apply_image(depth)
            image = np.concatenate([image, depth], -1)
        elif not self.rgb_on and self.depth_on:
            image = depth
            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            aug_input, transformation = T.apply_transform_gens(self.augmentations, aug_input)
            image, pan_seg_gt = aug_input.image, aug_input.sem_seg
            depth = transformation.apply_image(depth)
        elif self.rgb_on and not self.depth_on:
            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            aug_input, transformation = T.apply_transform_gens(self.augmentations, aug_input)
            image, pan_seg_gt = aug_input.image, aug_input.sem_seg

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
        dataset_dict.update(targets)
        h, w = image.shape[:2]

        # Load perturbed segmentation
        perturbed_segms = dataset_dict["perturbed_segmentation"]
        perturbed_masks = []
        for segm in perturbed_segms:
            mask = mask_util.decode(segm)
            mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)
            perturbed_masks.append(mask)
       
        # load gt for explicit error estimation
        if self.eee_mask_on:
            dataset_dict['tp_mask'] = torch.as_tensor(mask_util.decode(dataset_dict['tp_mask']).astype("long")).unsqueeze(0) # [1, H, W]
            dataset_dict['tn_mask'] = torch.as_tensor(mask_util.decode(dataset_dict['tn_mask']).astype("long")).unsqueeze(0)
            dataset_dict['fp_mask'] = torch.as_tensor(mask_util.decode(dataset_dict['fp_mask']).astype("long")).unsqueeze(0)
            dataset_dict['fn_mask'] = torch.as_tensor(mask_util.decode(dataset_dict['fn_mask']).astype("long")).unsqueeze(0)
        if self.eee_boundary_on:
            tp_boundary = torch.as_tensor(mask_util.decode(dataset_dict['tp_boundary']).astype("long")).unsqueeze(0)
            dataset_dict["tp_boundary"] = F.interpolate(tp_boundary.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze(0).long()
            tn_boundary = torch.as_tensor(mask_util.decode(dataset_dict['tn_boundary']).astype("long")).unsqueeze(0)
            dataset_dict["tn_boundary"] = F.interpolate(tn_boundary.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze(0).long()
            fp_boundary = torch.as_tensor(mask_util.decode(dataset_dict['fp_boundary']).astype("long")).unsqueeze(0)
            dataset_dict["fp_boundary"] = F.interpolate(fp_boundary.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze(0).long()
            fn_boundary = torch.as_tensor(mask_util.decode(dataset_dict['fn_boundary']).astype("long")).unsqueeze(0)
            dataset_dict["fn_boundary"] = F.interpolate(fn_boundary.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze(0).long()
        if self.offset_input_on:
            dataset_dict.update(self.perturbed_input_generator(perturbed_masks, h, w, h, w)) # [3, H, W]
        
        # print(h, w, ori_h, ori_w)
        # visualize
        # import matplotlib.pyplot as plt
        # cv2.imwrite('image.png', dataset_dict["image"].permute(1, 2, 0).numpy()[..., :3])
        # plt.imshow(dataset_dict["sem_seg"].numpy())
        # plt.savefig('sem_seg.png')
        # plt.imshow(dataset_dict["center"].numpy())
        # plt.savefig('center.png')
        # plt.imshow(dataset_dict["offset"].numpy()[0])
        # plt.savefig('offset_x.png')
        # plt.imshow(dataset_dict["offset"].numpy()[1])
        # plt.savefig('offset_y.png')
        # plt.imshow(dataset_dict["initial_pred_offset"].numpy()[0])
        # plt.savefig('initial_pred_center.png')
        # plt.imshow(dataset_dict["initial_pred_offset"].numpy()[1])
        # plt.savefig('initial_pred_offset.png')
        # plt.imshow(dataset_dict["tp_boundary"].numpy()[0])
        # plt.savefig('tp_boundary.png')

        perturbed_masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in perturbed_masks]
        if len(perturbed_masks) == 0:
            perturbed_masks = [torch.zeros((image.shape[0], image.shape[1]), dtype=torch.uint8)]
        dataset_dict["perturbed_masks"] = BitMasks(torch.stack(perturbed_masks))

        return dataset_dict