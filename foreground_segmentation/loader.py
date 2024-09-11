import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2
import json
import data_augmentation
from PIL import Image

NUM_VIEWS_PER_SCENE = 7

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation

###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


############# Synthetic Tabletop Object Dataset #############

class Tabletop_Object_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """


    def __init__(self, base_dir, train_or_test):
        self.base_dir = base_dir
        self.train_or_test = train_or_test

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '/*'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE
        print(f"Found {self.len} images in {self.base_dir} ({self.train_or_test})  ")

        self.name = 'TableTop'



    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = data_augmentation.random_color_warp(rgb_img)
        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        depth_img = data_augmentation.add_noise_to_depth(depth_img)

        # Compute xyz ordered point cloud
        # normalize 
        depth_img[depth_img < 0.3] = 0.3
        depth_img[depth_img > 1.5] = 1.5
        depth_img = (depth_img - 0.3) / (1.5 - 0.3)
        depth_img = np.expand_dims(depth_img, -1)

        return depth_img

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE

        # RGB image
        rgb_img_filename = scene_dir + f"/rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        depth_img_filename = scene_dir + f"/depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        depth_img = cv2.resize(depth_img, (640, 480))
        depth_img = self.process_depth(depth_img)
        
        # Labels
        foreground_labels_filename = scene_dir + f"/segmentation_{view_num:05d}.png"
        foreground_labels = imread_indexed(foreground_labels_filename)
        foreground_labels[foreground_labels>1] = 2
        foreground_labels = cv2.resize(foreground_labels, (640, 480), interpolation=cv2.INTER_NEAREST)
        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        depth_img = data_augmentation.array_to_tensor(depth_img) # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]

        return rgb_img, depth_img, foreground_labels



def get_TOD_train_dataloader(base_dir, batch_size=8, num_workers=4, shuffle=True):

    dataset = Tabletop_Object_Dataset(base_dir + 'TODv2/training_set/', 'train')

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_TOD_test_dataloader(base_dir, batch_size=8, num_workers=4, shuffle=False):

    dataset = Tabletop_Object_Dataset(base_dir + 'tabletop_dataset_v5_public/test_set/', 'test')

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)
