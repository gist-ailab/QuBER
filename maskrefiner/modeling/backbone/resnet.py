# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import (
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
)
from detectron2.modeling.backbone import Backbone

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn


class DeepLabStem(CNNBlockBase):
    """
    The DeepLab ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv3 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling.backbone.resnet import (
    Backbone,
    BasicBlock, 
    BottleneckBlock,
    DeformBottleneckBlock,
)

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv_1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv_1)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0, prefix=""):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                # [{prefix + "res2": 1, prefix + "res3": 2, prefix + "res4": 3, prefix + "res5": 4}.get(f, 0) for f in out_features]
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = prefix + "res" + str(i + 2) # adding underbar to avoid duplicate keys with RGB backbone
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        if prefix != "":
            self._out_features = [prefix + f for f in out_features]
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.
        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.
        Returns:
            list[CNNBlockBase]: a list of block module.
        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )
        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.
        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.
        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret


@BACKBONE_REGISTRY.register()
def build_resnet_deeplab_fusion_backbone(cfg, input_shape, prefix=""):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    offset_input_on = cfg.INPUT.OFFSET_INPUT_ON
    in_channels = 6 if offset_input_on else 3

    if cfg.MODEL.RESNETS.STEM_TYPE == "basic":
        stem = BasicStem(
            in_channels=in_channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    elif cfg.MODEL.RESNETS.STEM_TYPE == "deeplab":
        stem = DeepLabStem(
            in_channels=in_channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    else:
        raise ValueError("Unknown stem type: {}".format(cfg.MODEL.RESNETS.STEM_TYPE))

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid     = cfg.MODEL.RESNETS.RES5_MULTI_GRID
    # fmt: on
    assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
    assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        if stage_idx == 5:
            stage_kargs.pop("dilation")
            stage_kargs["dilation_per_block"] = [dilation * mg for mg in res5_multi_grid]
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, prefix=prefix).freeze(freeze_at)



class RGBDFusionBackbone(Backbone):


    def __init__(self, cfg, rgb_backbone, depth_backbone):

        # get the attribute of rgb_backbone
        super(RGBDFusionBackbone, self).__init__()

        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        # get the all attributes of depth_backbone (include submodules), and add the prefix "depth_"
        # this is for the loading of the pre-trained model
      

        self.fusion_strategy = cfg.MODEL.BACKBONE.FUSION_STRATEGY
        self.num_fusion_layers = cfg.MODEL.BACKBONE.NUM_FUSION_LAYERS
        #!TODO: support more various backbone
        self.out_features = cfg.MODEL.RESNETS.OUT_FEATURES
        channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
        self.fusion_layers = {}
        for out_feature in self.out_features:
            self.fusion_layers[out_feature] = nn.Sequential()
            if self.fusion_strategy == "concat":
                self.fusion_layers[out_feature].add_module("conv", nn.Conv2d(channels[out_feature] * 2, channels[out_feature], kernel_size=1, stride=1, padding=0))
                self.fusion_layers[out_feature].add_module("gn", nn.GroupNorm(32, channels[out_feature]))
                self.fusion_layers[out_feature].add_module("relu", nn.ReLU(inplace=True)),
            for i in range(self.num_fusion_layers):
                if out_feature == "res5":
                    continue
                self.fusion_layers[out_feature].add_module("conv{}".format(i), nn.Conv2d(channels[out_feature], channels[out_feature], kernel_size=3, stride=1, padding=1))
                self.fusion_layers[out_feature].add_module("gn{}".format(i), nn.GroupNorm(32, channels[out_feature]))
                self.fusion_layers[out_feature].add_module("relu{}".format(i), nn.ReLU(inplace=True))
            self.add_module("fusion_{}".format(out_feature), self.fusion_layers[out_feature])
        
        self._out_features = self.rgb_backbone._out_features
        self._out_feature_channels = self.rgb_backbone._out_feature_channels
        self._out_feature_strides = self.rgb_backbone._out_feature_strides



    def forward(self, x):
        rgb_x = x[:, :3, :, :]
        depth_x = x[:, 3:6, :, :]
        offset_x = x[:, 6:, :, :]
        rgb_x = torch.cat([rgb_x, offset_x], dim=1)
        depth_x = torch.cat([depth_x, offset_x], dim=1)
        rgb_out = self.rgb_backbone(rgb_x)
        depth_out = self.depth_backbone(depth_x)
        for k in rgb_out.keys():
            if self.fusion_strategy == "concat":
                rgb_out[k] = torch.cat([rgb_out[k], depth_out["depth_" + k]], dim=1)
            elif self.fusion_strategy == "add":
                rgb_out[k] = rgb_out[k] + depth_out["depth_" + k]
            rgb_out[k] = self.fusion_layers[k](rgb_out[k])
        return rgb_out
          

@BACKBONE_REGISTRY.register()
def build_resnet_deeplab_rgbd_fusion_backbone(cfg, input_shape):
   
    depth_on = cfg.INPUT.DEPTH_ON
    assert depth_on, "depth_on must be True for RGBD fusion backbone"

    rgb_backbone = build_resnet_deeplab_fusion_backbone(cfg, input_shape)
    depth_backbone = build_resnet_deeplab_fusion_backbone(cfg, input_shape, prefix="depth_")
    rgbd_fusion_backbone = RGBDFusionBackbone(cfg, rgb_backbone, depth_backbone)
    return rgbd_fusion_backbone