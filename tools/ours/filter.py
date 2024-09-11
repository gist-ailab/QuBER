import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import random
import time
import os
from pycocotools.coco import COCO
from pycocotools import mask as m

from detectron2.utils.visualizer import Visualizer
from perturbation_utils import *
from tqdm import tqdm
import datetime

import json
from tqdm import tqdm


uoais_sim_path = '/SSDe/seunghyeok_back/mask-refiner/detectron2_datasets/UOAIS-Sim'
split = 'val'
perturbed_coco_anno_path = os.path.join(uoais_sim_path, 'annotations', 'uoais_sim_panoptic_{}_perturbed.json'.format(split))

with open(perturbed_coco_anno_path, 'r') as f:
    coco_anno = json.load(f)

print(coco_anno.keys())
print(coco_anno['segments_info'])