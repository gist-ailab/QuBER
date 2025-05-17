import os
import imageio
import cv2
import detectron2
import torch
import time
import numpy as np


from quber.predictor import QuBERPredictor
from quber.fg_segm import lmffNet
from quber.modeling.mask_refiner.post_processing import get_panoptic_segmentation
from preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor, compute_xyz

W = 640 
H = 480

os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


class QuBER():

    def __init__(self, config_file, weights_file, dataset='OSD'):

        self.refiner_predictor = QuBERPredictor(config_file, weights_file= weights_file)
        self.lmffnet = lmffNet()
        self.dataset = dataset


    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        rgb_img = cv2.imread(rgb_path)
        if 'npy' in depth_path:
            depth_img = np.load(depth_path)
        else:
            depth_img = imageio.imread(depth_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        zero_depth = np.where(depth_img == 0)
        if 'npy' in depth_path:
            depth_img = normalize_depth(depth_img, 0.25, 1.5)
        else:
            depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        if initial_masks.dtype == np.bool:
            initial_masks = np.uint8(initial_masks) * 255

        start_time = time.time()
        output = self.refiner_predictor.predict(rgb_img, depth_img, initial_masks)[0]
        if "instances" not in output.keys():
            refined_masks = []
        else:
            refined_instances = output['instances'].to('cpu')
            refined_masks = refined_instances.pred_masks.detach().cpu().numpy()
        time_elapsed = time.time() - start_time
        fg_mask = self.lmffnet.predict(rgb_path, depth_path)
        filt_masks = []
        for refined_mask in refined_masks:
            if np.sum(np.bitwise_and(refined_mask, fg_mask)) / np.sum(refined_mask) > 0.3:
                filt_masks.append(refined_mask)
        time_elapsed = time.time() - start_time
        if self.dataset == 'OCID':
            # The methods using the xyz images (e.g RICE, UOIS, UCN, MSMFormer) automatically filter out the zero-depth pixels.
            # DoPose dataset's labels are 0 for zero-depth pixels.
            # Thus, we need to filter out the zero-depth pixels for these two datasets.
            filt_masks2 = []
            for refined_mask in filt_masks:
                refined_mask[zero_depth] = False
                filt_masks2.append(refined_mask)
            filt_masks = filt_masks2
            refined_masks = np.asarray(filt_masks)


        return refined_masks, output, time_elapsed, fg_mask # [N, H, W], bool


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True
