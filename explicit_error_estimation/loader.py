import os
import cv2
import numpy as np
import torch
import random
import json
from pycocotools import mask as m

from scipy.stats import multivariate_normal
from scipy.ndimage import center_of_mass

import albumentations as A
import torch.utils.data as data
import glob
from torchvision.transforms import Normalize
import imageio
import random
from util import *

from pycocotools.coco import COCO

import pyfastnoisesimd as fns

def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm) 
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth

def inpaint_depth(depth, factor=1, kernel_size=3, dilate=True):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """
    
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W//factor, H//factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth

def perlin_noise(frequency, width, height):

    noise = fns.Noise()
    noise.NoiseType = 2 # perlin noise
    noise.frequency = frequency
    result = noise.genAsGrid(shape=[height, width], start=[0,0])
    return result

def PerlinDistortion(image, width, height):
    """
    """
    # sample distortion parameters from noise vector
    fx = np.random.uniform(0.0001, 0.5)
    fy = np.random.uniform(0.0001, 0.5)
    fz = np.random.uniform(0.01, 0.5)
    wxy = np.random.uniform(0, 20)
    wz = np.random.uniform(0, 0.01)
    cnd_x = wxy * perlin_noise(fx, width, height)
    cnd_y = wxy * perlin_noise(fy, width, height)
    cnd_z = wz * perlin_noise(fz, width, height)

    cnd_h = np.array(list(range(height)))
    cnd_h = np.expand_dims(cnd_h, -1)
    cnd_h = np.repeat(cnd_h, width, -1)
    cnd_w = np.array(list(range(width)))
    cnd_w = np.expand_dims(cnd_w, 0)
    cnd_w = np.repeat(cnd_w, height, 0)

    noise_cnd_h = np.int16(cnd_h + cnd_x)
    noise_cnd_h = np.clip(noise_cnd_h, 0, (height - 1))
    noise_cnd_w = np.int16(cnd_w + cnd_y)
    noise_cnd_w = np.clip(noise_cnd_w, 0, (width - 1))

    new_img = image[(noise_cnd_h, noise_cnd_w)]
    new_img = new_img = new_img + cnd_z
    return new_img.astype(np.float32)

class UOAISSimDataset(data.Dataset):

    def __init__(self, cfg, train=True):

        print("Initializing data loader: {} for {}".format(cfg["train_set"], "train" if train else "test"))
        self.data_root = os.environ["DETECTRON2_DATASETS"]

        dataset = cfg["train_set"] if train else cfg["val_set"]
        if dataset == "uoais_sim_train":
            self.dataset_path = os.path.join(self.data_root, "UOAIS-Sim/train")
            self.coco_anns_path = os.path.join(self.data_root, "UOAIS-Sim/annotations/coco_anns_uoais_sim_train_perturbed.json")
        elif dataset == "uoais_sim_val":
            self.dataset_path = os.path.join(self.data_root, "UOAIS-Sim/val")
            self.coco_anns_path = os.path.join(self.data_root, "UOAIS-Sim/annotations/coco_anns_uoais_sim_val_perturbed.json")
        else:
            print("Unknown dataset: {}".format(dataset))
            raise NotImplementedError
        self.coco_anno = COCO(self.coco_anns_path)
        self.img_ids = self.coco_anno.getImgIds()
        self.img_size = cfg["img_size"]

        self.train = train
        self.perturbed_input_offset_generator = PerturbedInputOffsetGenerator()

        # transformation
        train_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.augmentations.transforms.ColorJitter(),
            A.augmentations.transforms.ChannelShuffle(),
            A.augmentations.transforms.RandomGamma(),
            A.augmentations.transforms.ImageCompression(),
            A.augmentations.HorizontalFlip(p=0.5),
            A.augmentations.RandomSizedCrop(min_max_height=(int(self.img_size[1]*0.5), int(self.img_size[1]*1.0)), height=self.img_size[1], width=self.img_size[0], p=0.5),
        ],
        additional_targets={'gt_masks': 'mask', 'perturbed_masks': 'mask', 'depth': 'mask'}
        )
        test_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(self.img_size[1], self.img_size[0])
        ],
        additional_targets={'gt_masks': 'mask', 'perturbed_masks': 'mask', 'depth': 'mask'}
        )
        if train:
            self.transform = train_transform
        else: 
            self.transform = test_transform
        self.color_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        img_info = self.coco_anno.loadImgs(img_id)[0]
        rgb_path = os.path.join(self.dataset_path, img_info["file_name"])
        img = cv2.imread(rgb_path) # [H, W, C]
        depth_path = os.path.join(self.dataset_path, img_info["depth_file_name"])
        depth = imageio.imread(depth_path) # [H, W]

        ann_ids = self.coco_anno.getAnnIds(imgIds=img_id)
        anns = self.coco_anno.loadAnns(ann_ids)
        gt_masks = np.array([self.coco_anno.annToMask(ann) for ann in anns]) # [N, H, W]
        gt_masks = np.transpose(gt_masks, (1, 2, 0)) # [H, W, N]
        perturbed_masks = img_info["perturbed_segmentation"]
        perturbed_masks = np.array([m.decode(perturbed_mask) for perturbed_mask in perturbed_masks])
        perturbed_masks = np.transpose(perturbed_masks, (1, 2, 0)) # [H, W, N]
        augmented = self.transform(image=img, gt_masks=gt_masks, perturbed_masks=perturbed_masks, depth=depth)
        image = augmented['image']
        gt_masks = augmented['gt_masks'].transpose(2, 0, 1) # [N, H, W]
        perturbed_masks = augmented['perturbed_masks'].transpose(2, 0, 1) # [N, H, W]
        depth = augmented['depth']
        depth = PerlinDistortion(depth, self.img_size[0], self.img_size[1])
        depth[depth>15000] = 15000
        depth[depth<2500] = 2500
        depth = (depth - 2500) / (15000 - 2500)
        depth = np.expand_dims(depth, axis=-1)
        depth = torch.tensor(depth.transpose(2, 0, 1), dtype=torch.float32)[0:1, :, :]

        input_offset = self.perturbed_input_offset_generator(perturbed_masks)
        input_fg_mask = masks_to_fg_mask(perturbed_masks)
        input_boundary = masks_to_boundary(perturbed_masks)

        gt_fg_mask = masks_to_fg_mask(gt_masks)
        gt_boundary = masks_to_boundary(gt_masks)

        # compute true positive, true negative, false positive masks
        tp_mask = np.logical_and(gt_fg_mask.astype(bool), input_fg_mask.astype(bool)).astype(np.uint8)
        tn_mask = np.logical_and(gt_fg_mask.astype(bool), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)
        fp_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), input_fg_mask.astype(bool)).astype(np.uint8)
        fn_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)

        tp_boundary = np.logical_and(gt_boundary.astype(bool), input_boundary.astype(bool)).astype(np.uint8)
        tn_boundary = np.logical_and(gt_boundary.astype(bool), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)
        fp_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), input_boundary.astype(bool)).astype(np.uint8)
        fn_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)
        

        ## visualization
        # input_offset = input_offset.numpy().transpose(1, 2, 0) 
        # input_offset[:, :, 0] = (input_offset[:, :, 0] - input_offset[:, :, 0].min()) / (input_offset[:, :, 0].max() - input_offset[:, :, 0].min())
        # input_fg_mask = input_fg_mask[np.newaxis, :, :].transpose(1, 2, 0).repeat(3, axis=2)
        # input_boundary = input_boundary[np.newaxis, :, :].transpose(1, 2, 0).repeat(3, axis=2)
        # gt_fg_mask = gt_fg_mask[np.newaxis, :, :].transpose(1, 2, 0).repeat(3, axis=2)
        # gt_boundary = gt_boundary[np.newaxis, :, :].transpose(1, 2, 0).repeat(3, axis=2)
        # tp_mask = tp_mask[np.newaxis, :, :].transpose(1, 2, 0) # green
        # tn_mask = tn_mask[np.newaxis, :, :].transpose(1, 2, 0) # blue
        # fp_mask = fp_mask[np.newaxis, :, :].transpose(1, 2, 0) # red
        # mask_vis = np.concatenate([tn_mask, tp_mask, fp_mask], axis=-1)
        # tp_boundary = tp_boundary[np.newaxis, :, :].transpose(1, 2, 0) # green
        # tn_boundary = tn_boundary[np.newaxis, :, :].transpose(1, 2, 0) # blue
        # fp_boundary = fp_boundary[np.newaxis, :, :].transpose(1, 2, 0) # red
        # boundary_vis = np.concatenate([tn_boundary, tp_boundary, fp_boundary], axis=-1)
        # vis = np.vstack([
        #     np.hstack([image, gt_fg_mask*255, input_fg_mask*255, mask_vis*255]), 
        #     np.hstack([np.uint8(input_offset*255), gt_boundary*255, input_boundary*255, boundary_vis*255])
        # ])

        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255
        image = self.color_normalize(image)

        # change to tensor
        input_fg_mask = torch.tensor(input_fg_mask, dtype=torch.float32).unsqueeze(0)
        input_boundary = torch.tensor(input_boundary, dtype=torch.float32).unsqueeze(0)
        gt_fg_mask = torch.tensor(gt_fg_mask, dtype=torch.float32).unsqueeze(0)
        gt_boundary = torch.tensor(gt_boundary, dtype=torch.float32).unsqueeze(0)
        tp_mask = torch.tensor(tp_mask, dtype=torch.float32).unsqueeze(0)
        tn_mask = torch.tensor(tn_mask, dtype=torch.float32).unsqueeze(0)
        fp_mask = torch.tensor(fp_mask, dtype=torch.float32).unsqueeze(0)
        fn_mask = torch.tensor(fn_mask, dtype=torch.float32).unsqueeze(0)
        tp_boundary = torch.tensor(tp_boundary, dtype=torch.float32).unsqueeze(0)
        tn_boundary = torch.tensor(tn_boundary, dtype=torch.float32).unsqueeze(0)
        fp_boundary = torch.tensor(fp_boundary, dtype=torch.float32).unsqueeze(0)
        fn_boundary = torch.tensor(fn_boundary, dtype=torch.float32).unsqueeze(0)

        # change gt to 2 channel (background, foreground)
        # gt_fg_mask = torch.cat([1-gt_fg_mask, gt_fg_mask], dim=0)
        # gt_boundary = torch.cat([1-gt_boundary, gt_boundary], dim=0)
        # tn_mask = torch.cat([1-tn_mask, tn_mask], dim=0)
        # fp_mask = torch.cat([1-fp_mask, fp_mask], dim=0)
        # tn_boundary = torch.cat([1-tn_boundary, tn_boundary], dim=0)
        # fp_boundary = torch.cat([1-fp_boundary, fp_boundary], dim=0)
        # tp_mask = torch.cat([1-tp_mask, tp_mask], dim=0)
        # tp_boundary = torch.cat([1-tp_boundary, tp_boundary], dim=0)
        return {
            "input_rgb": image, # [3, H, W]
            'input_depth': depth,
            "input_offset": input_offset, #[1, H, W]
            "input_fg_mask": input_fg_mask,
            "input_boundary": input_boundary,
            "gt_fg_mask": gt_fg_mask,
            "gt_boundary": gt_boundary,
            "tp_mask": tp_mask,
            "tn_mask": tn_mask,
            "fp_mask": fp_mask,
            "fn_mask": fn_mask,
            "tp_boundary": tp_boundary,
            "tn_boundary": tn_boundary,
            "fp_boundary": fp_boundary,
            "fn_boundary": fn_boundary,
        }

        
    def __len__(self):
        return len(self.img_ids)


class OSDDataset(data.Dataset):

    def __init__(self, cfg):

        print("Initializing OSD dataset")
        self.data_root = os.environ["DETECTRON2_DATASETS"]
        osd_path = os.path.join(self.data_root, "OSD-0.2-depth")

        self.rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(osd_path)))
        self.depth_paths = sorted(glob.glob("{}/disparity/*.png".format(osd_path)))
        self.anno_paths = sorted(glob.glob("{}/annotation/*.png".format(osd_path)))
        self.perturbed_paths = sorted(glob.glob("{}/uoais/*.png".format(osd_path)))
        assert len(self.rgb_paths) == len(self.anno_paths)
        assert len(self.rgb_paths) == len(self.perturbed_paths)

        self.img_size = cfg["img_size"]
        self.perturbed_input_offset_generator = PerturbedInputOffsetGenerator()
        self.color_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __getitem__(self, idx):

        image = cv2.imread(self.rgb_paths[idx])
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]))

        depth = imageio.imread(self.depth_paths[idx])
        depth = cv2.resize(depth, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        depth = normalize_depth(depth)
        depth = inpaint_depth(depth)/255
        depth = depth.astype(np.float32)
        depth = depth.transpose(2, 0, 1)[0:1, :, :]
        input_depth = torch.tensor(depth, dtype=torch.float32)


        perturbed_mask = cv2.imread(self.perturbed_paths[idx])
        perturbed_mask = cv2.resize(perturbed_mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)[:, :, 0]
        perturbed_ids = np.unique(perturbed_mask).tolist()
        perturbed_ids.remove(0)
        perturbed_masks = []
        for id in perturbed_ids:
            mask = np.uint8(np.where(perturbed_mask == id, 1, 0))
            perturbed_masks.append(mask)
        
        gt_mask = cv2.imread(self.anno_paths[idx])
        gt_mask = cv2.resize(gt_mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)[:, :, 0]
        gt_ids = np.unique(gt_mask).tolist()
        gt_ids.remove(0)
        gt_masks = []
        for id in gt_ids:
            mask = np.uint8(np.where(gt_mask == id, 1, 0))
            gt_masks.append(mask)

        input_offset = self.perturbed_input_offset_generator(perturbed_masks)
        input_fg_mask = masks_to_fg_mask(perturbed_masks)
        input_boundary = masks_to_boundary(perturbed_masks)

        gt_fg_mask = masks_to_fg_mask(gt_masks)
        gt_boundary = masks_to_boundary(gt_masks)

        # compute true positive, true negative, false positive masks
        tp_mask = np.logical_and(gt_fg_mask.astype(bool), input_fg_mask.astype(bool)).astype(np.uint8)
        tn_mask = np.logical_and(gt_fg_mask.astype(bool), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)
        fp_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), input_fg_mask.astype(bool)).astype(np.uint8)
        fn_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)

        tp_boundary = np.logical_and(gt_boundary.astype(bool), input_boundary.astype(bool)).astype(np.uint8)
        tn_boundary = np.logical_and(gt_boundary.astype(bool), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)
        fp_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), input_boundary.astype(bool)).astype(np.uint8)
        fn_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)

        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255
        image = self.color_normalize(image)

        # change to tensor
        input_fg_mask = torch.tensor(input_fg_mask, dtype=torch.float32).unsqueeze(0)
        input_boundary = torch.tensor(input_boundary, dtype=torch.float32).unsqueeze(0)
        gt_fg_mask = torch.tensor(gt_fg_mask, dtype=torch.float32).unsqueeze(0)
        gt_boundary = torch.tensor(gt_boundary, dtype=torch.float32).unsqueeze(0)
        tp_mask = torch.tensor(tp_mask, dtype=torch.float32).unsqueeze(0)
        tn_mask = torch.tensor(tn_mask, dtype=torch.float32).unsqueeze(0)
        fp_mask = torch.tensor(fp_mask, dtype=torch.float32).unsqueeze(0)
        fn_mask = torch.tensor(fn_mask, dtype=torch.float32).unsqueeze(0)
        tp_boundary = torch.tensor(tp_boundary, dtype=torch.float32).unsqueeze(0)
        tn_boundary = torch.tensor(tn_boundary, dtype=torch.float32).unsqueeze(0)
        fp_boundary = torch.tensor(fp_boundary, dtype=torch.float32).unsqueeze(0)
        fn_boundary = torch.tensor(fn_boundary, dtype=torch.float32).unsqueeze(0)

        return {
            "input_rgb": image, # [3, H, W]
            "input_depth": input_depth, # [1, H, W]
            "input_offset": input_offset, #[1, H, W]
            "input_fg_mask": input_fg_mask,
            "input_boundary": input_boundary,
            "gt_fg_mask": gt_fg_mask,
            "gt_boundary": gt_boundary,
            "tp_mask": tp_mask,
            "tn_mask": tn_mask,
            "fp_mask": fp_mask,
            "fn_mask": fn_mask,
            "tp_boundary": tp_boundary,
            "tn_boundary": tn_boundary,
            "fp_boundary": fp_boundary,
            "fn_boundary": fn_boundary,
        }


    def __len__(self):
        return len(self.rgb_paths)


class OCIDDataset(data.Dataset):

    def __init__(self, cfg):

        print("Initializing OSD dataset")
        self.data_root = os.environ["DETECTRON2_DATASETS"]

        # load dataset
        rgb_paths = []
        depth_paths = []
        anno_paths = []
        # load ARID20
        print("... load dataset [ ARID20 ]")
        data_root = os.path.join(self.data_root, "OCID-dataset", "ARID20")
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                seq_list = sorted(os.listdir(os.path.join(data_root, dir_1, dir_2)))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        rgb_path = os.path.join(data_root, dir_1, dir_2, seq, "rgb", data_name)
                        rgb_paths.append(rgb_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, seq, "label", data_name)
                        anno_paths.append(anno_path)
        # load YCB10
        print("... load dataset [ YCB10 ]")
        data_root = os.path.join(self.data_root, "OCID-dataset", "YCB10")
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        c_c_m = ["cuboid", "curved", "mixed"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                for dir_3 in c_c_m:
                    seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                    for seq in seq_list:
                        data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                        if not os.path.isdir(data_dir): continue
                        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                        for data_name in data_list:
                            rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                            rgb_paths.append(rgb_path)
                            depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                            depth_paths.append(depth_path)
                            anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                            anno_paths.append(anno_path)
        # load ARID10
        print("... load dataset [ ARID10 ]")
        data_root = os.path.join(self.data_root, "OCID-dataset", "ARID10")
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        c_c_m = ["box", "curved", "fruits", "mixed", "non-fruits"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                for dir_3 in c_c_m:
                    seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                    for seq in seq_list:
                        data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                        if not os.path.isdir(data_dir): continue
                        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                        for data_name in data_list:
                            rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                            rgb_paths.append(rgb_path)
                            depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                            depth_paths.append(depth_path)
                            anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                            anno_paths.append(anno_path)
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print("Evaluation on OCID dataset: {} rgbs, {} depths, {} visible_masks".format(
                        len(rgb_paths), len(depth_paths), len(anno_paths)))
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.anno_paths = anno_paths

        self.img_size = cfg["img_size"]
        self.perturbed_input_offset_generator = PerturbedInputOffsetGenerator()
        self.color_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __getitem__(self, idx):

        image = cv2.imread(self.rgb_paths[idx])
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]))

        depth = imageio.imread(self.depth_paths[idx])
        depth = cv2.resize(depth, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        depth = normalize_depth(depth)
        depth = inpaint_depth(depth)/255
        depth = depth.astype(np.float32)
        depth = depth.transpose(2, 0, 1)[0:1, :, :]
        input_depth = torch.tensor(depth, dtype=torch.float32)


        perturbed_mask = cv2.imread(self.perturbed_paths[idx])
        perturbed_mask = cv2.resize(perturbed_mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)[:, :, 0]
        perturbed_ids = np.unique(perturbed_mask).tolist()
        perturbed_ids.remove(0)
        perturbed_masks = []
        for id in perturbed_ids:
            mask = np.uint8(np.where(perturbed_mask == id, 1, 0))
            perturbed_masks.append(mask)
        
        gt_mask = cv2.imread(self.anno_paths[idx])
        gt_mask = cv2.resize(gt_mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)[:, :, 0]
        gt_ids = np.unique(gt_mask).tolist()
        gt_ids.remove(0)
        gt_masks = []
        for id in gt_ids:
            mask = np.uint8(np.where(gt_mask == id, 1, 0))
            gt_masks.append(mask)

        input_offset = self.perturbed_input_offset_generator(perturbed_masks)
        input_fg_mask = masks_to_fg_mask(perturbed_masks)
        input_boundary = masks_to_boundary(perturbed_masks)

        gt_fg_mask = masks_to_fg_mask(gt_masks)
        gt_boundary = masks_to_boundary(gt_masks)

        # compute true positive, true negative, false positive masks
        tp_mask = np.logical_and(gt_fg_mask.astype(bool), input_fg_mask.astype(bool)).astype(np.uint8)
        tn_mask = np.logical_and(gt_fg_mask.astype(bool), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)
        fp_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), input_fg_mask.astype(bool)).astype(np.uint8)
        fn_mask = np.logical_and(np.logical_not(gt_fg_mask.astype(bool)), np.logical_not(input_fg_mask.astype(bool))).astype(np.uint8)

        tp_boundary = np.logical_and(gt_boundary.astype(bool), input_boundary.astype(bool)).astype(np.uint8)
        tn_boundary = np.logical_and(gt_boundary.astype(bool), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)
        fp_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), input_boundary.astype(bool)).astype(np.uint8)
        fn_boundary = np.logical_and(np.logical_not(gt_boundary.astype(bool)), np.logical_not(input_boundary.astype(bool))).astype(np.uint8)

        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255
        image = self.color_normalize(image)

        # change to tensor
        input_fg_mask = torch.tensor(input_fg_mask, dtype=torch.float32).unsqueeze(0)
        input_boundary = torch.tensor(input_boundary, dtype=torch.float32).unsqueeze(0)
        gt_fg_mask = torch.tensor(gt_fg_mask, dtype=torch.float32).unsqueeze(0)
        gt_boundary = torch.tensor(gt_boundary, dtype=torch.float32).unsqueeze(0)
        tp_mask = torch.tensor(tp_mask, dtype=torch.float32).unsqueeze(0)
        tn_mask = torch.tensor(tn_mask, dtype=torch.float32).unsqueeze(0)
        fp_mask = torch.tensor(fp_mask, dtype=torch.float32).unsqueeze(0)
        fn_mask = torch.tensor(fn_mask, dtype=torch.float32).unsqueeze(0)
        tp_boundary = torch.tensor(tp_boundary, dtype=torch.float32).unsqueeze(0)
        tn_boundary = torch.tensor(tn_boundary, dtype=torch.float32).unsqueeze(0)
        fp_boundary = torch.tensor(fp_boundary, dtype=torch.float32).unsqueeze(0)
        fn_boundary = torch.tensor(fn_boundary, dtype=torch.float32).unsqueeze(0)

        return {
            "input_rgb": image, # [3, H, W]
            "input_depth": input_depth, # [1, H, W]
            "input_offset": input_offset, #[1, H, W]
            "input_fg_mask": input_fg_mask,
            "input_boundary": input_boundary,
            "gt_fg_mask": gt_fg_mask,
            "gt_boundary": gt_boundary,
            "tp_mask": tp_mask,
            "tn_mask": tn_mask,
            "fp_mask": fp_mask,
            "fn_mask": fn_mask,
            "tp_boundary": tp_boundary,
            "tn_boundary": tn_boundary,
            "fp_boundary": fp_boundary,
            "fn_boundary": fn_boundary,
        }


    def __len__(self):
        return len(self.rgb_paths)