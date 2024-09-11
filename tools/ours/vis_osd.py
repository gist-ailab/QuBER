import os 
import glob
import cv2
import imageio 
import numpy as np
import open3d as o3d
from tqdm import tqdm

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


dataset_path = 'detectron2_datasets/OSD-0.2-depth'

# load dataset
rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))
assert len(rgb_paths) == len(depth_paths)
assert len(rgb_paths) == len(anno_paths)

BACKGROUND_LABEL = 0
W, H = 640, 480
metrics_all = []
iou_masks = 0
num_inst_all = 0 # number of all instances
num_inst_mat = 0 # number of matched instance

for img_idx, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, anno_paths)):

    assert os.path.basename(rgb_path) == os.path.basename(depth_path)
    assert os.path.basename(rgb_path) == os.path.basename(anno_path)
    print(rgb_path, depth_path, anno_path)
    file_name = os.path.basename(rgb_path)
    # load rgb and depth
    rgb_img = cv2.imread(rgb_path)
    depth_img = imageio.imread(depth_path)
    depth_img = normalize_depth(depth_img)
    anno = imageio.imread(anno_path)
    anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
    labels_anno = np.unique(anno)
    labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]

    # draw anno on rgb and depth
    rgb_anno = rgb_img.copy()
    depth_anno = depth_img.copy()
    for label in labels_anno:
        color = np.random.randint(0, 255, 3)
        rgb_anno[anno == label] = color * 0.3 + rgb_anno[anno == label] * 0.7
        depth_anno[anno == label] = color * 0.3 + depth_anno[anno == label] * 0.7

    vis = np.hstack([rgb_img, depth_img, rgb_anno, depth_anno])
    cv2.imwrite('vis/{}'.format(file_name), vis)

