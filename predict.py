from maskrefiner.predictor import MaskReformerPredictor
from maskrefiner.config import add_panoptic_deeplab_config


# config_file = 'configs/uoais-sim/instance-segmentation/maskformer2_R50_bs2.yaml'
# config_file = 'configs/uoais-sim/instance-segmentation/maskreformer_R50_bs2_simple_concat.yaml'
# dataset_name = 'uoais_sim_instance_val'

# predictor = MaskReformerPredictor(config_file, dataset_name)
# predictor.predict(0)
# predictor.predict_all()



import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import random
import glob
# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from maskrefiner.config import add_maskformer2_config, add_mask_encoder_config


class Predictor():
    def __init__(self):
        cfg = get_cfg()
        add_mask_encoder_config(cfg)
        add_deeplab_config(cfg)
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file("configs/uoais-sim/instance-segmentation/Panoptic-DeepLab-LR1e-3.yaml")
        cfg.MODEL.WEIGHTS = 'output/uoais-sim/instance-segmentation/Panoptic-DeepLab-LR1e-4/model_0019999.pth'
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("uoais_sim_val_panoptic")


    def predict(self, image):
        im = cv2.imread(image)
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        result = instance_result[:, :, ::-1]
        cv2.imwrite('out.png', result)

predictor = Predictor()
img_paths = glob.glob('/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/UOAIS-Sim/val/bin/color/*.png')
img_path = random.choice(list(img_paths))
predictor.predict(img_path)
