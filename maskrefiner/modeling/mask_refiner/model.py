# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.utils.registry import Registry

from monai.losses import DiceLoss
from .post_processing import get_panoptic_segmentation

__all__ = ["MaskRefiner", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""

class DeepLabBCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabBCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.BCEWithLogitsLoss(
            weight=weight, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        logits = logits.squeeze(1)
        labels = labels.float()
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()

@META_ARCH_REGISTRY.register()
class MaskRefiner(nn.Module):
    """
    Main class for panoptic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        #!TODO: add more fusion strategies
        #!TODO: build fusion module inside the build_backbone
        self.offset_input_on = cfg.INPUT.OFFSET_INPUT_ON

        self.depth_on = cfg.INPUT.DEPTH_ON

        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV

        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

        self.eee_mask_on = cfg.MODEL.INS_EMBED_HEAD.EEE_MASK_ON
        self.eee_boundary_on = cfg.MODEL.INS_EMBED_HEAD.EEE_BOUNDARY_ON
        self.eee_postprocess_on = cfg.MODEL.INS_EMBED_HEAD.EEE_POST_PROCESS_ON

        self.error_type = cfg.MODEL.INS_EMBED_HEAD.ERROR_TYPE



    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:
                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )
        images = ImageList.from_tensors(images, size_divisibility)
        

        if self.offset_input_on:
            initial_pred_offset = [x["initial_pred_offset"].to(self.device) for x in batched_inputs]
            initial_pred_offset = ImageList.from_tensors(initial_pred_offset, size_divisibility).tensor
            # concat with image
            input = torch.cat([images.tensor, initial_pred_offset], dim=1)
        else:
            input = images.tensor
        features = self.backbone(input)

        losses = {}

        if "sem_seg" in batched_inputs[0] and "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            foreground_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            foreground_targets = ImageList.from_tensors(
                foreground_targets, size_divisibility
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                foreground_weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                foreground_weights = ImageList.from_tensors(foreground_weights, size_divisibility).tensor
            else:
                foreground_weights = None


            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
            if self.eee_mask_on:
                if self.error_type == "e3":
                    eee_mask_targets = [torch.cat([x["tp_mask"].to(self.device), 
                                                    x["tn_mask"].to(self.device),
                                                    x["fp_mask"].to(self.device),
                                                    x["fn_mask"].to(self.device)], dim=0) for x in batched_inputs]
                elif self.error_type == "e2":
                    eee_mask_targets = [torch.cat(
                        [(x["tp_mask"] + x["tn_mask"]).to(self.device),
                        (x["fp_mask"] + x["fn_mask"]).to(self.device)], dim=0) for x in batched_inputs]     
                elif self.error_type == "e33":
                    eee_mask_targets = [torch.cat(
                        [(x["tp_mask"] + x["tn_mask"]).to(self.device),
                        x["fp_mask"].to(self.device), 
                        x["fn_mask"].to(self.device)], dim=0) for x in batched_inputs]     
                elif self.error_type == "e32":
                    eee_mask_targets = [torch.cat(
                        [x["fp_mask"].to(self.device), 
                        x["fn_mask"].to(self.device)], dim=0) for x in batched_inputs]    

                eee_mask_targets = ImageList.from_tensors(eee_mask_targets, size_divisibility).tensor
            else:
                eee_mask_targets = None
            if self.eee_boundary_on:
                if self.error_type == "e3":
                    eee_boundary_targets = [torch.cat([x["tp_boundary"].to(self.device),
                                                        x["tn_boundary"].to(self.device),
                                                        x["fp_boundary"].to(self.device),
                                                        x["fn_boundary"].to(self.device)], dim=0) for x in batched_inputs]
                elif self.error_type == "e2":
                    eee_boundary_targets = [torch.cat(
                        [(x["tp_boundary"] + x["tn_boundary"]).to(self.device),
                        (x["fp_boundary"] + x["fn_boundary"]).to(self.device)], dim=0) for x in batched_inputs]    
                elif self.error_type == "e33":
                    eee_boundary_targets = [torch.cat(
                        [(x["tp_boundary"] + x["tn_boundary"]).to(self.device),
                        x["fp_boundary"].to(self.device), 
                        x["fn_boundary"].to(self.device)], dim=0) for x in batched_inputs]     
                elif self.error_type == "e32":
                    eee_boundary_targets = [torch.cat(
                        [x["fp_boundary"].to(self.device), 
                        x["fn_boundary"].to(self.device)], dim=0) for x in batched_inputs]     
                eee_boundary_targets = ImageList.from_tensors(eee_boundary_targets, size_divisibility).tensor
            else:
                eee_boundary_targets = None

        else:
            foreground_targets = None
            foreground_weights = None

            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

            eee_mask_targets = None
            eee_boundary_targets = None

        output_dict, loss_dict = self.ins_embed_head(
            features, 
            foreground_targets, foreground_weights,
            center_targets, center_weights, 
            offset_targets, offset_weights, 
            eee_mask_targets, eee_boundary_targets
        )
        losses.update(loss_dict)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for idx, (input_per_image, image_size) in enumerate(zip(
            batched_inputs, images.image_sizes
        )):
            height = batched_inputs[idx]["height"]
            width = batched_inputs[idx]["width"]
            foreground_result = output_dict["foreground"][idx]
            r = sem_seg_postprocess(foreground_result, image_size, height, width)
            center_result = output_dict["center"][idx]
            c = sem_seg_postprocess(center_result, image_size, height, width)
            offset_result = output_dict["offset"][idx]
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            
            # # visualize the center and offset
            # import matplotlib.pyplot as plt
            # import cv2
            # # clear
            # plt.clf()
            # print(c.shape, o.shape)
            # plt.imshow(c[0].detach().cpu().numpy())
            # plt.savefig('tmp/center_{}.png'.format(idx))
            # plt.imshow(o[0].detach().cpu().numpy())
            # plt.savefig('tmp/offset_{}.png'.format(idx))
            
            m, b = None, None
            if self.eee_mask_on:
                m = output_dict["eee_mask"][idx]
                m = sem_seg_postprocess(m, image_size, height, width)
            if self.eee_boundary_on:
                b = output_dict["eee_boundary"][idx]
                b = sem_seg_postprocess(b, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.sigmoid().round(),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            # For test-time augmentation.
            # processed_results[-1]["center"] = c
            # processed_results[-1]["offset"] = o
            if self.eee_boundary_on:
                processed_results[-1]["eee_boundary"] = b
            if self.eee_mask_on:
                processed_results[-1]["eee_mask"] = m

            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = r.sigmoid().squeeze(0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            idx = 0
            instances = []
            panoptic_image_cpu = panoptic_image.cpu().detach().numpy()
            # print("panoptic_image", np.unique(panoptic_image_cpu), panoptic_image_cpu.shape)
            for panoptic_label in np.unique(panoptic_image_cpu):
                if panoptic_label == -1:
                    continue
                pred_class = panoptic_label // self.meta.label_divisor - 1
                instance = Instances((height, width))
                # Evaluation code takes continuous id starting from 0
                instance.pred_classes = torch.tensor(
                    [pred_class], device=panoptic_image.device
                )
                mask = panoptic_image == panoptic_label
                # cv2.imwrite('tmp/mask_{}.png'.format(idx), mask.cpu().numpy().astype(np.uint8) * 255)
                idx += 1
                instance.pred_masks = mask.unsqueeze(0)
                # Average semantic probability
                # sem_scores = semantic_prob[pred_class, ...]
                # import cv2
                # cv2.imwrite("tmp/sem_seg_{}.png".format(idx), (semantic_prob.round() * 255).cpu().numpy().astype(np.uint8))
                sem_scores = semantic_prob.clone()
                sem_scores = torch.mean(sem_scores[mask])
                # Center point probability
                mask_indices = torch.nonzero(mask).float()
                center_y, center_x = (
                    torch.mean(mask_indices[:, 0]),
                    torch.mean(mask_indices[:, 1]),
                )
                center_scores = c[0, int(center_y.item()), int(center_x.item())]
                # Confidence score is semantic prob * center prob.
                instance.scores = torch.tensor(
                    [sem_scores * center_scores], device=panoptic_image.device
                )
                # Get bounding boxes
                instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                instances.append(instance)
            if len(instances) > 0:
                processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


class SinglePredictionHead(nn.Module):
    def __init__(self, in_channels, head_channels, use_bias, norm, use_depthwise_separable_conv):
        super().__init__()
        if use_depthwise_separable_conv:
            self.head = DepthwiseSeparableConv2d(
                in_channels,
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=get_norm(norm, in_channels),
                activation1=F.relu,
                norm2=get_norm(norm, head_channels),
                activation2=F.relu,
            )
            weight_init.c2_xavier_fill(self.head)
        else:
            self.head = nn.Sequential(
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, in_channels),
                    activation=F.relu,
                ),
                Conv2d(
                    in_channels,
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])


    def forward(self, x):
        x = self.head(x)
        return x
    
class SinglePredictor(nn.Module):

    def __init__(self, head_channels, out_channels):
        super().__init__()
        self.predictor = Conv2d(head_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        return self.predictor(x)

class FusionLayers(nn.Module):

    def __init__(self, in_channels, out_channels, num_fusion_layers, fusion_strategy, norm):
        super().__init__()
        self.fusion_layers = nn.ModuleList()
        # add 1x1 conv to reduce channels
        self.fusion_layers.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                norm=get_norm("BN", out_channels),
                activation=F.relu,
            ))

        for i in range(num_fusion_layers):
            self.fusion_layers.append(
                Conv2d(
                     out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm=get_norm("BN", out_channels),
                    activation=F.relu,
                )
            )
            weight_init.c2_xavier_fill(self.fusion_layers[i])

    def forward(self, x):
        for layer in self.fusion_layers:
            x = layer(x)
        return x
    

@INS_EMBED_BRANCHES_REGISTRY.register()
class MaskRefinerInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        foreground_loss_weight: float,
        foreground_loss_type: str,
        foreground_loss_top_k: float,
        center_loss_weight: float,
        offset_loss_weight: float,
        eee_mask_on: bool,
        eee_mask_loss_type: str,
        eee_mask_loss_weight: float,
        eee_boundary_on: bool,
        eee_boundary_loss_type: str,
        eee_boundary_loss_weight: float,
        hierarchical_fusion_on: bool,
        hierarchy: List[List[str]],
        num_fusion_layers: int,
        fusion_strategy: str,
        fusion_target: List[str],
        error_type: str,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""

        self.foreground_pred_head = SinglePredictionHead(
            decoder_channels[0], head_channels, use_bias, norm, self.use_depthwise_separable_conv
        )
        self.foreground_predictor = SinglePredictor(head_channels, 1)

        self.center_pred_head = SinglePredictionHead(
            decoder_channels[0], head_channels, use_bias, norm, self.use_depthwise_separable_conv
        )
        self.center_predictor = SinglePredictor(head_channels, 1)

        self.offset_pred_head = SinglePredictionHead(
            decoder_channels[0], head_channels, use_bias, norm, self.use_depthwise_separable_conv
        )
        self.offset_predictor = SinglePredictor(head_channels, 2)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

        self.foreground_loss_weight = foreground_loss_weight
        if foreground_loss_type == "cross_entropy":
            self.foreground_loss = nn.CrossEntropyLoss(reduction="mean")
        elif foreground_loss_type == "hard_pixel_mining":
            self.foreground_loss = DeepLabBCE(top_k_percent_pixels=foreground_loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % foreground_loss_type)

        self.eee_mask_on = eee_mask_on
        self.eee_boundary_on = eee_boundary_on
        self.eee_mask_loss_weight = eee_mask_loss_weight
        self.eee_boundary_loss_weight = eee_boundary_loss_weight
        self.error_type = error_type
        if self.eee_mask_on:
            self.eee_mask_pred_head = SinglePredictionHead(
                decoder_channels[0], head_channels, use_bias, norm, self.use_depthwise_separable_conv
            )
            if error_type == 'e3':
                self.eee_mask_predictor = SinglePredictor(head_channels, 4)
            elif error_type == 'e2' or error_type == 'e32':
                self.eee_mask_predictor = SinglePredictor(head_channels, 2)
            elif error_type == 'e33':
                self.eee_mask_predictor = SinglePredictor(head_channels, 3)
            if eee_mask_loss_type == "cross_entropy":
                self.eee_mask_loss = nn.CrossEntropyLoss(reduction="mean")
            elif eee_mask_loss_type == "dice":
                self.eee_mask_loss = DiceLoss(softmax=True)
        if self.eee_boundary_on:
            self.eee_boundary_pred_head = SinglePredictionHead(
                decoder_channels[0], head_channels, use_bias, norm, self.use_depthwise_separable_conv
            )
            if error_type == 'e3':
                self.eee_boundary_predictor = SinglePredictor(head_channels, 4)
            elif error_type == 'e2' or error_type == 'e32':
                self.eee_boundary_predictor = SinglePredictor(head_channels, 2)
            elif error_type == 'e33':
                self.eee_boundary_predictor = SinglePredictor(head_channels, 3)
            if eee_boundary_loss_type == "cross_entropy":
                self.eee_boundary_loss = nn.CrossEntropyLoss(reduction="mean")  
            elif eee_boundary_loss_type == "dice":
                self.eee_boundary_loss = DiceLoss(softmax=True)

        self.hierarchical_fusion_on = hierarchical_fusion_on
        self.hierarchy = hierarchy
        self.num_fusion_layers = num_fusion_layers
        self.fusion_strategy = fusion_strategy
        self.fusion_target = fusion_target

        if self.hierarchical_fusion_on:
            self.fusion_layers = {}
            for i, h in enumerate(self.hierarchy):
                if i == 0:
                    continue
                in_channels = decoder_channels[0]
                if "feat" in self.fusion_target:
                    in_channels += head_channels * len(self.hierarchy[i-1])
                if "pred" in self.fusion_target:
                    for output_name in self.hierarchy[i-1]:
                        if "eee" in output_name:
                            if self.error_type == "e3":
                                in_channels += 4
                            elif self.error_type == "e33":
                                in_channels += 3
                            elif error_type == 'e2' or error_type == 'e32':
                                in_channels += 2
                        elif "offset" in output_name:
                            in_channels += 2
                        else:
                            in_channels += 1
                self.fusion_layers[i] = FusionLayers(
                    in_channels=in_channels,
                    out_channels=decoder_channels[0],
                    num_fusion_layers=self.num_fusion_layers,
                    fusion_strategy=self.fusion_strategy,
                    norm = norm,
                )
                self.add_module("fusion_layers_{}".format(i), self.fusion_layers[i])

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            foreground_loss_weight=cfg.MODEL.INS_EMBED_HEAD.FOREGROUND_LOSS_WEIGHT,
            foreground_loss_type=cfg.MODEL.INS_EMBED_HEAD.FOREGROUND_LOSS_TYPE,
            foreground_loss_top_k=cfg.MODEL.INS_EMBED_HEAD.FOREGROUND_LOSS_TOP_K,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
            eee_mask_on=cfg.MODEL.INS_EMBED_HEAD.EEE_MASK_ON,
            eee_mask_loss_type=cfg.MODEL.INS_EMBED_HEAD.EEE_MASK_LOSS_TYPE,
            eee_mask_loss_weight=cfg.MODEL.INS_EMBED_HEAD.EEE_MASK_LOSS_WEIGHT,
            eee_boundary_on=cfg.MODEL.INS_EMBED_HEAD.EEE_BOUNDARY_ON,
            eee_boundary_loss_type=cfg.MODEL.INS_EMBED_HEAD.EEE_BOUNDARY_LOSS_TYPE,
            eee_boundary_loss_weight=cfg.MODEL.INS_EMBED_HEAD.EEE_BOUNDARY_LOSS_WEIGHT,
            hierarchical_fusion_on = cfg.MODEL.INS_EMBED_HEAD.HIERARCHICAL_FUSION_ON,
            hierarchy = cfg.MODEL.INS_EMBED_HEAD.HIERARCHY,
            num_fusion_layers = cfg.MODEL.INS_EMBED_HEAD.NUM_FUSION_LAYERS,
            fusion_strategy = cfg.MODEL.INS_EMBED_HEAD.FUSION_STRATEGY,
            fusion_target = cfg.MODEL.INS_EMBED_HEAD.FUSION_TARGET,
            error_type = cfg.MODEL.INS_EMBED_HEAD.ERROR_TYPE,
        )
        return ret

    def forward(
        self,
        features,
        forground_targets=None,
        foreground_weights=None,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
        eee_mask_targets=None,
        eee_boundary_targets=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        output_dict = self.layers(features)

        if self.training:
            loss_dict = {}
            loss_dict.update(self.foreground_losses(output_dict["foreground"], forground_targets, foreground_weights))
            loss_dict.update(self.center_losses(output_dict["center"], center_targets, center_weights))
            loss_dict.update(self.offset_losses(output_dict["offset"], offset_targets, offset_weights))
            if self.eee_mask_on:
                predictions = F.interpolate(
                    output_dict["eee_mask"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                loss_dict["loss_eee_mask"] = self.eee_mask_loss(predictions, eee_mask_targets)
            if self.eee_boundary_on:
                predictions = F.interpolate(
                    output_dict["eee_boundary"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                loss_dict["loss_eee_boundary"] = self.eee_boundary_loss(predictions, eee_boundary_targets)
            return output_dict, loss_dict
        else:
            output_dict["foreground"] = F.interpolate(
                output_dict["foreground"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            output_dict["center"] = F.interpolate(
                output_dict["center"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            output_dict["offset"] = (
                F.interpolate(
                    output_dict["offset"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            if self.eee_mask_on:
                output_dict["eee_mask"] = F.interpolate(
                    output_dict["eee_mask"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            if self.eee_boundary_on:
                output_dict["eee_boundary"] = F.interpolate(
                    output_dict["eee_boundary"], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            return output_dict, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)

        feature_dict = {}
        if not self.hierarchical_fusion_on:
            if self.eee_mask_on:
                feature_dict["eee_mask"] = self.eee_mask_pred_head(y)
            if self.eee_boundary_on:
                feature_dict["eee_boundary"] = self.eee_boundary_pred_head(y)
            feature_dict["foreground"] = self.foreground_pred_head(y)
            feature_dict["center"] = self.center_pred_head(y)
            feature_dict["offset"] = self.offset_pred_head(y)
            foreground = self.foreground_predictor(feature_dict["foreground"])
            center = self.center_predictor(feature_dict["center"])
            offset = self.offset_predictor(feature_dict["offset"])
            output_dict = {"foreground": foreground, "center": center, "offset": offset}
            if self.eee_mask_on:
                eee_mask = self.eee_mask_predictor(feature_dict["eee_mask"])
                output_dict["eee_mask"] = eee_mask
            if self.eee_boundary_on:
                eee_boundary = self.eee_boundary_predictor(feature_dict["eee_boundary"])
                output_dict["eee_boundary"] = eee_boundary
        else:
            # if not self.eee_boundary_on or not self.eee_boundary_on:
            #     raise NotImplementedError

            # self.hierarchy = [["eee_mask", "eee_boundary"], ["foreground", "center", "offset"]]
            output_dict = {}
            for i, h in enumerate(self.hierarchy):
                if i == 0:
                    for key in h:
                        feature_dict[key] = getattr(self, key + "_pred_head")(y)
                        output_dict[key] = getattr(self, key + "_predictor")(feature_dict[key])

                else:
                    y_prime = y
                    if "feat" in self.fusion_target:
                        # fuse i-1 level features
                        for prev_key in self.hierarchy[i-1]:
                            y_prime = torch.cat([y_prime, feature_dict[prev_key]], dim=1)
                    if "pred" in self.fusion_target:
                        for prev_key in self.hierarchy[i-1]:
                            output = output_dict[prev_key]
                            if "eee" in prev_key:
                                output = output.softmax(dim=1)
                            else:
                                output = output.sigmoid()
                            y_prime = torch.cat([y_prime, output], dim=1)
                    for key in h:
                        feature_dict[key] = getattr(self, key + "_pred_head")(self.fusion_layers[i](y_prime))
                        output_dict[key] = getattr(self, key + "_predictor")(feature_dict[key])

        return output_dict

    def foreground_losses(self, predictions, targets, weights=None):
        # in original panoptic deeplab, sem_seg is used to segments stuff
        # in our case, we use sem_seg to segment things
        # thus, we need to invert the semantic segmentation
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.foreground_loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.foreground_loss_weight}
        return losses

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses