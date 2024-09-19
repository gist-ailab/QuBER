import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from pycocotools import mask as m

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

# import Mask2Former project
from maskrefiner import add_maskformer2_config, add_mask_refiner_config, add_panoptic_deeplab_config, PerturbedPanopticDatasetMapper
from maskrefiner.data.dataset_mappers.target_generator import *

import typing
from collections import defaultdict

import tabulate
from torch import nn

def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(model: nn.Module, max_depth: int = 3) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table. It looks like this:

    ::

        | name                            | #elements or shape   |
        |:--------------------------------|:---------------------|
        | model                           | 37.9M                |
        |  backbone                       |  31.5M               |
        |   backbone.fpn_lateral3         |   0.1M               |
        |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
        |    backbone.fpn_lateral3.bias   |    (256,)            |
        |   backbone.fpn_output3          |   0.6M               |
        |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output3.bias    |    (256,)            |
        |   backbone.fpn_lateral4         |   0.3M               |
        |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
        |    backbone.fpn_lateral4.bias   |    (256,)            |
        |   backbone.fpn_output4          |   0.6M               |
        |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output4.bias    |    (256,)            |
        |   backbone.fpn_lateral5         |   0.5M               |
        |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
        |    backbone.fpn_lateral5.bias   |    (256,)            |
        |   backbone.fpn_output5          |   0.6M               |
        |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output5.bias    |    (256,)            |
        |   backbone.top_block            |   5.3M               |
        |    backbone.top_block.p6        |    4.7M              |
        |    backbone.top_block.p7        |    0.6M              |
        |   backbone.bottom_up            |   23.5M              |
        |    backbone.bottom_up.stem      |    9.4K              |
        |    backbone.bottom_up.res2      |    0.2M              |
        |    backbone.bottom_up.res3      |    1.2M              |
        |    backbone.bottom_up.res4      |    7.1M              |
        |    backbone.bottom_up.res5      |    14.9M             |
        |    ......                       |    .....             |

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        # if x > 1e8:
        #     return "{:.5f}G".format(x / 1e9)
        if x > 1e5:
            return "{:.5f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.5f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab




class Predictor():

    def __init__(self, config_file, dataset_name):

        self.config_file = config_file
        print('predicting with config file: ', config_file)
        self.cfg = get_cfg()

        add_deeplab_config(self.cfg)
        add_panoptic_deeplab_config(self.cfg)
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = config_file.replace('.yaml', '/model_final.pth').replace('configs', 'output')

        self.model = build_model(self.cfg)
        self.model.eval()
        if len(self.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)


        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        mapper = PerturbedInstanceDatasetMapper(self.cfg, False)
        self.loader = build_detection_test_loader(self.cfg, dataset_name, mapper=mapper)
        self.n_images = len(self.loader)
        # print('Number of images: {}'.format(self.n_images))
        self.metadata = MetadataCatalog.get(dataset_name)
        self.output_dir = '{}/visualization'.format(self.config_file[:-5].replace("configs", "output"))
        os.makedirs(self.output_dir, exist_ok=True)
        print('Output directory: {}'.format(self.output_dir))

    def predict(self, idx):
        
        inputs = [self.loader.dataset[idx]]
        outputs = self.model(inputs)[0]
        im = inputs[0]["image"].permute(1, 2, 0).numpy()
        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        result = instance_result[:, :, ::-1]
        if 'maskreformer' in self.config_file:
            perturbed_masks = inputs[0]['perturbed_segmentation']
            perturbed_masks = [m.decode(perturbed_mask) for perturbed_mask in perturbed_masks]
            visualizer = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
            viz = visualizer.overlay_instances(masks=perturbed_masks, alpha=1.0)
            perturbed_mask_viz = viz.get_image()[:, :, ::-1]
            result = np.hstack((result, perturbed_mask_viz))
        cv2.imwrite(self.output_dir +'/{}.png'.format(idx), result)

    def predict_all(self):
        for idx in tqdm(range(self.n_images)):
            self.predict(idx)


class MaskRefinerPredictor():

    def __init__(self, config_file, dataset_name='uoais_sim_val_panoptic', weights_file=None):
        # add configs and load config file and weights
        self.cfg = get_cfg()
        add_panoptic_deeplab_config(self.cfg)
        add_mask_refiner_config(self.cfg)
        self.cfg.merge_from_file(config_file)
     
        # load model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(self.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        if weights_file is not None:
            self.cfg.MODEL.WEIGHTS = config_file.replace('.yaml', '/{}'.format(weights_file)).replace('configs', 'output')
        else:
            self.cfg.MODEL.WEIGHTS = config_file.replace('.yaml', '/model_final.pth').replace('configs', 'output')
        self.cfg.MODEL.WEIGHTS = '/SSDe/seunghyeok_back/mask-refiner/model_0219999.pth'
        print('Loading weights file: ', self.cfg.MODEL.WEIGHTS )
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        # set ipnuts
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        mapper = PerturbedPanopticDatasetMapper(self.cfg, False)
        self.loader = build_detection_test_loader(self.cfg, dataset_name, mapper=mapper)
        self.n_images = len(self.loader)
        print('Number of images: {}'.format(self.n_images))
        self.metadata = MetadataCatalog.get(dataset_name)
        self.output_dir = '{}/visualization'.format(config_file[:-5].replace("configs", "output"))
        os.makedirs(self.output_dir, exist_ok=True)
        print('Output directory: {}'.format(self.output_dir))

        self.sigma = 10
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

        self.depth_on = self.cfg.INPUT.DEPTH_ON
        self.rgb_on = self.cfg.INPUT.RGB_ON
        print('Num of parameters:',  parameter_count_table(self.model, 10))

        import detectron2.model_zoo as model_zoo
        # freeze_layers = self.cfg.MODEL.BACKBONE.FREEZE_LAYERS
        # weight_path = self.cfg.MODEL.BACKBONE.WEIGHTS
        # if weight_path != "":
        #     print("Loading pretrained weights: ", weight_path)
        #     pretrained_model = model_zoo.get(weight_path, trained=True)
        #     pretrained_backbone = pretrained_model
        #     for name, parameter in self.model.named_parameters():
        #         if 'depth' in name:
        #             continue
        #         freeze = False
        #         for target_name in freeze_layers:
        #             if target_name in name:
        #                 freeze = True
        #                 break
        #         freeze = True
        #         if freeze:
        #             parameter.requires_grad = False
        #             # iterate over all layers in the pretrained weights
        #             for pretrained_name, pretrained_parameter in pretrained_backbone.named_parameters():
        #                 pretrained_name = pretrained_name.replace("bottom_up.", "")
        #                 if pretrained_name in name:
        #                     print("Load and freeze pretrained layer: {} from {}".format(name, pretrained_name))
        #                     try:
        #                         parameter.data.copy_(pretrained_parameter.data)
        #                     except:
        #                         print("Load pretrained layer {} failed".format(pretrained_name))



    def predict(self, rgb_img, depth_img=None, perturbed_masks=None,):
        
        
        inputs = {}
        height, width = rgb_img.shape[:2]
        inputs["height"] = height
        inputs["width"] = width
        inputs["image"] = torch.as_tensor(np.ascontiguousarray(rgb_img.transpose(2, 0, 1)))

        if not self.rgb_on and self.depth_on:
            depth = torch.as_tensor(np.ascontiguousarray(depth_img.transpose(2, 0, 1)))
            inputs["image"] = depth

        if self.rgb_on and self.depth_on:
            depth = torch.as_tensor(np.ascontiguousarray(depth_img.transpose(2, 0, 1)))
            inputs["image"] = torch.cat((inputs["image"], depth), dim=0)

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
        
        offsets = np.stack([center, offset[0], offset[1]], axis=0)
        # visualize
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(center)
        # axs[1].imshow(offset[0])
        # axs[2].imshow(offset[1])
        # plt.savefig('offsets.png')
        offsets = torch.as_tensor(offsets.astype(np.float32))
        inputs["initial_pred_offset"] = offsets
        output = self.model([inputs])
        return output


    def draw(self, image):
        im = cv2.imread(image)
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        result = instance_result[:, :, ::-1]
        cv2.imwrite('out.png', result)
