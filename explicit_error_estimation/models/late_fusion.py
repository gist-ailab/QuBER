
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.manet.decoder import MAnetDecoder
from segmentation_models_pytorch.decoders.linknet.decoder import LinknetDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.decoders.pspnet.decoder import PSPDecoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.pan.decoder import PANDecoder


# some private imports for create_model function
from typing import Optional as _Optional
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationHead,
)
import torch 
import torch.nn as nn

class LateFusionEncoderDecoder(nn.Module):

    def __init__(
        self,
        device = torch.device("cpu"),
        encoder_name: str = "resnet34",
        encoder_weights: _Optional[str] = "imagenet",
        encoder_depth: int = 5,
        encoder_output_stride: int = 32,
        inputs = ["offset", "fg_mask", "boundary"],
        heads = ["mask", "boundary"],
        targets = ["tp", "tn", "fp"],
        decoder_class = PSPDecoder,
        decoder_dim: int = 512,
        **kwargs,
    ):
        super().__init__()

        self.inputs = inputs

        # encoder
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.encoders = {}

        for input in inputs:
            if input == "offset":
                in_channels = 3
            if input == "fg_mask":
                in_channels = 1
            if input == "boundary":
                in_channels = 1
            if input == "rgb":
                in_channels = 3
            if input == "depth":
                in_channels = 1

            self.encoders[input] = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
                output_stride=encoder_output_stride,
            ).to(device)
            self.add_module('encoder_' + input, self.encoders[input])

        # fusion layer
        self.encoder_channels = self.encoders[input].out_channels[1:]
        self.fusion_layers = {}
        for idx in range(len(self.encoder_channels)):
            self.fusion_layers['fusion_layer_' + str(idx)] = nn.Conv2d(
                in_channels=self.encoder_channels[idx] * len(self.inputs),
                out_channels=self.encoder_channels[idx],
                kernel_size=1,
                stride=1,
                padding=0,
            ).to(device)
            self.add_module('fusion_layer_' + str(idx), self.fusion_layers['fusion_layer_' + str(idx)])


        # decoder
        try:
            self.decoder = decoder_class(
                encoder_channels = self.encoder_channels,
                out_channels = decoder_dim,
            ).to(device)
        except TypeError:
            self.decoder = decoder_class(
                in_channels = self.encoder_channels[-1],
                out_channels = decoder_dim,
            ).to(device)


        # segmentation heads
        self.heads = heads
        self.targets = targets
        self.segmentation_heads = {}
        if decoder_class in [DeepLabV3Decoder, DeepLabV3PlusDecoder]:
            upsampling = 4
        elif decoder_class == PSPDecoder:
            upsampling = 8
        for head in heads:
            args = {
                'in_channels': decoder_dim,
                'out_channels': len(targets),
                'kernel_size': 1,
            }
            if decoder_class in [DeepLabV3Decoder, DeepLabV3PlusDecoder, PSPDecoder]:
                args['upsampling'] = upsampling
            self.segmentation_heads[head] = \
                SegmentationHead(**args).to(device)
            self.add_module(head, self.segmentation_heads[head])
        self.decoder_class = decoder_class


    def forward(self, data):
        encoder_features_all = []
        for input_name in self.inputs:
            input = data['input_' + input_name]
            encoder_features = self.encoders[input_name](input)[1:]
            encoder_features_all.append(encoder_features)

        fused_features_all = []
        for idx in range(len(self.encoder_channels)):
            features_idx = []
            for features in encoder_features_all:
                features_idx.append(features[idx])
            features_idx = torch.cat(features_idx, dim=1)
            fused_features_all.append(self.fusion_layers['fusion_layer_' + str(idx)](features_idx))
        decoder_output = self.decoder(*fused_features_all)
        outputs = {}
        for head in self.heads:
            outputs[head] = self.segmentation_heads[ head](decoder_output)
        return outputs

def create_late_fusion_model(
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    encoder_output_stride: int = 32,
    decoder_name: str = "pspdecoder",
    decoder_dim = 256,
    encoder_depth: int = 5,
    inputs = ["offset", "fg_mask", "boundary"],
    heads = ["mask", "boundary"],
    targets = ["tp", "tn", "fp"],
    device = torch.device("cpu"),
    **kwargs,
) -> torch.nn.Module:
    """Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    decoder_archs = [
        UnetDecoder, UnetPlusPlusDecoder, MAnetDecoder, LinknetDecoder, FPNDecoder,
        DeepLabV3Decoder, DeepLabV3PlusDecoder, PSPDecoder, PANDecoder,
    ]
    decoder_archs_dict = {a.__name__.lower(): a for a in decoder_archs}
    try:
        decoder_class = decoder_archs_dict[decoder_name.lower()]
    except KeyError:
        raise KeyError(
            "Wrong decoder architecture type `{}`. Available options are: {}".format(
                decoder_name,
                list(decoder_archs_dict.keys()),
            )
        )

    return LateFusionEncoderDecoder(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_depth = encoder_depth,
        encoder_output_stride = encoder_output_stride,
        inputs = inputs,
        heads = heads,
        targets = targets,
        decoder_class = decoder_class,
        decoder_dim = decoder_dim,
        device = device,
        **kwargs,
    )