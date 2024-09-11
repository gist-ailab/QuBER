import torch
import copy
import numpy as np
import cv2
import os
# from adet.config import get_cfg
# from adet.utils.post_process import detector_postprocess, DefaultPredictor
import segmentation_models_pytorch as smp
import sys
uoais_path = '/SSDe/seunghyeok_back/uoais'
sys.path.append(uoais_path)
# from foreground_segmentation.model import Context_Guided_Network

def get_initial_metric(targets, heads, best=False):
    default = []
    if best:
        default = 0
    metrics = {}
    for target in targets:
        for head in heads:
            metrics[head + '_iou_all'] = copy.deepcopy(default)
            metrics[head + '_iou'] = copy.deepcopy(default)
            # metrics[target + "_" + head + '_f1'] = copy.deepcopy(default)
            # metrics[target + "_" + head + '_accuracy'] = copy.deepcopy(default)
            # metrics[target + "_" + head + '_precision'] = copy.deepcopy(default)
            # metrics[target + "_" + head + '_recall'] = copy.deepcopy(default)
    return metrics    

def compute_metrics(preds, data, metrics, targets):
    
    for head, pred in preds.items():
        pred = torch.argmax(pred, dim=1, keepdim=True)
        # convert it to the stack of binary images
        pred = torch.cat([pred == i for i in range(len(targets))], dim=1).to(torch.int64)
        pred = pred.detach()
        gt = torch.cat([data[x + '_' + head] for x in targets], dim=1)
        gt = gt.detach().to(torch.int64)
        tp, fp, fn, tn = smp.metrics.get_stats(pred, gt, mode='multiclass', num_classes=4)
        iou_all = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

        tp, fp, fn, tn = smp.metrics.get_stats(pred-1, gt-1, mode='multiclass', num_classes=3, ignore_index=-1)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        # f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
        # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
        # precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
        # recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
        
        metrics[head + "_iou_all"].append(iou_all.item())
        metrics[head + "_iou"].append(iou.item())
        # metrics[head + "_f1"].append(f1.item())
        # metrics[head + "_accuracy"].append(accuracy.item())
        # metrics[head + "_precision"].append(precision.item())
        # metrics[head + "_recall"].append(recall.item())
    return metrics

def get_average_metrics(metrics):
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    return metrics


def masks_to_fg_mask(masks):
    # masks = [N, H, W], numpy array
    fg_mask = np.zeros_like(masks[0])
    for mask in masks:
        fg_mask += mask
    fg_mask = fg_mask > 0
    return fg_mask.astype(np.uint8)


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def masks_to_boundary(masks, dilation_ratio=0.01):
    # masks = [N, H, W], numpy array
    fg_mask = masks_to_fg_mask(masks)
    boundary = np.zeros_like(fg_mask)
    for mask in masks:
        boundary += mask_to_boundary(mask, dilation_ratio=dilation_ratio)
    boundary = boundary > 0
    return boundary.astype(np.uint8)

class PerturbedInputOffsetGenerator(object):
    """
    Generates training targets for Panoptic-DeepLab.
    """

    def __init__(
        self,
        sigma=10,
        ignore_stuff_in_offset=True,
        small_instance_area=4096,
        small_instance_weight=3,
        ignore_crowd_in_semantic=False,
    ):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            thing_ids: Set, a set of ids from contiguous category ids belonging
                to thing categories.
            sigma: the sigma for Gaussian kernel.
            ignore_stuff_in_offset: Boolean, whether to ignore stuff region when
                training the offset branch.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for
                small instances.
            ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in
                semantic segmentation branch, crowd region is ignored in the original
                TensorFlow implementation.
        """
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def __call__(self, perturbed_masks):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/main/datasets/prepare_panoptic_fpn.py#L18  # noqa
        Args:
            panoptic: numpy.array, panoptic label, we assume it is already
                converted from rgb image by panopticapi.utils.rgb2id.
            segments_info (list[dict]): see detectron2 documentation of "Use Custom Datasets".
        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(H, W).
                - center_points: List, center coordinates, with tuple
                    (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - sem_seg_weights: Tensor, loss weight for semantic prediction,
                    shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction,
                    shape=(H, W), used as weights for center regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
        """
        height, width = perturbed_masks[0].shape

        center = np.zeros((height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )
       
        for perturbed_mask in perturbed_masks:
            # print(np.unique(panoptic), seg["id"])
            # find instance center
            mask_index = np.where(perturbed_mask != 0)
            if len(mask_index[0]) == 0:
                # the instance is completely cropped
                continue

            # Find instance area
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])

            # generate center heatmap
            y, x = int(round(center_y)), int(round(center_x))
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            # start and end indices in default Gaussian image
            gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
            gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

            # start and end indices in center heatmap image
            center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
            center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
            center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                center[center_y0:center_y1, center_x0:center_x1],
                self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
            )

            # generate offset (2, h, w) -> (y-dir, x-dir)
            # normalize by width and height (-1~1)
            offset[0][mask_index] = (center_y - y_coord[mask_index]) / height
            offset[1][mask_index] = (center_x - x_coord[mask_index]) / width


        # import imageio
        # from detectron2.utils.visualizer import Visualizer
        # _offset = np.where(offset == 0, 0, offset)
        # imageio.imwrite('offset_maps.png', np.hstack([center, _offset[0], _offset[1]]))
        # visualizer= Visualizer(np.zeros_like(center), scale=1.0)
        # mask_vis = visualizer.overlay_instances(masks=perturbed_masks, alpha=1.0).get_image()
        # imageio.imwrite('mask_vis.png', mask_vis)
        # print("center_pts", np.min(center), np.max(center))
        # print("x_offset", np.min(offset[0]), np.max(offset[0]))
        # print("y_offset", np.min(offset[1]), np.max(offset[1]))
        # exit()
        
        offsets = np.stack([center, offset[0], offset[1]], axis=0)
        offsets = torch.as_tensor(offsets.astype(np.float32))
        return offsets

def torch_to_img(tensor, text=None, binary=True, rgb=False, offset=False):
    if tensor.shape[1] == 2:
        # if torch.unique(tensor).shape[0] == 2:
        #     tensor = tensor[:, 1:2, :, :]
        # else:
        tensor = torch.sigmoid(tensor)
        # compress 2 channels to 1 by argmax
        tensor = torch.argmax(tensor, dim=1, keepdim=True)
            # invert
    if rgb:
        # unnormalize , 
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        tensor = tensor * 255
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    img = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    if offset:
        img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].min()) / (img[:, :, 0].max() - img[:, :, 0].min())
        img = img * 255
    img = np.array(img.copy(), dtype=np.uint8) 
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if binary:
        img = img * 255
    if text is not None:
        img = cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def visualize_inference(data, preds, targets, logdir, file_name,):
    
    input_rgb = torch_to_img(data["input_rgb"], text="input_rgb", binary=False, rgb=True)
    input_offset = torch_to_img(data["input_offset"], text="input_offset", offset=True, binary=False)
    input_fg_mask = torch_to_img(data["input_fg_mask"], text="input_fg_mask")
    input_boundary = torch_to_img(data["input_boundary"], text="input_boundary")
    gt_fg_mask = torch_to_img(data["gt_fg_mask"], text="gt_mask")
    gt_boundary = torch_to_img(data["gt_boundary"], text="gt_boundary")
    imgs = [np.vstack([input_rgb, input_offset]),
            np.vstack([input_fg_mask, input_boundary]),
            np.vstack([gt_fg_mask, gt_boundary])]
    for head, pred in preds.items():
        gt_vis = np.zeros_like(input_rgb)
        pred = torch.argmax(pred, dim=1, keepdim=True)[0]
        pred = torch.cat([pred == i for i in range(len(targets))], dim=0)
        pred = pred.detach()
        pred_vis = np.zeros_like(input_rgb)
        if 'tn' in targets:
            gt_vis[:, :, 0:1] = data['tn_' + head].detach()[0].cpu().numpy().transpose(1, 2, 0) * 255
            pred_vis[:, :, 0] = pred[targets.index('tn')].cpu().numpy() * 255
        if 'tp' in targets:
            gt_vis[:, :, 1:2] = data['tp_' + head].detach()[0].cpu().numpy().transpose(1, 2, 0) * 255
            pred_vis[:, :, 1] = pred[targets.index('tp')].cpu().numpy() * 255
        if 'fp' in targets:
            gt_vis[:, :, 2:3] = data['fp_' + head].detach()[0].cpu().numpy().transpose(1, 2, 0) * 255
            pred_vis[:, :, 2] = pred[targets.index('fp')].cpu().numpy() * 255
        gt_vis = cv2.putText(gt_vis, "gt_" + head, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
        pred_vis = cv2.putText(pred_vis, "pred_" + head, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
        img = np.vstack([gt_vis, pred_vis])
        imgs.append(img)
    imgs = np.hstack(imgs)
    cv2.imwrite(os.path.join(logdir, file_name + '.png'), imgs)



BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]

def load_uoais(config_file):
    print("Use UOAIS to detect instances")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(uoais_path, cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def load_cgnet(cgnet_weight_path):
    print("Use foreground segmentation model (CG-Net) to filter out background instances")
    checkpoint = torch.load(os.path.join(cgnet_weight_path))
    fg_model = Context_Guided_Network(classes=2, in_channel=4)
    fg_model.load_state_dict(checkpoint['model'])
    fg_model.cuda()
    fg_model.eval()
    return fg_model

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

def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ unnormalize the input depth (0 ~ 255) and return depth image (mm)
    Args:
        depth([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.
    Returns:
        [np.float]: depth array [H, W] (mm) 
    """
    depth = np.float32(depth) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
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


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

