# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)

_PREDEFINED_SPLITS = {
    "tod_v2_train_panoptic": (
        "TODv2",
        "TODv2/annotations/tod_v2_train_panoptic_perturbated.json",
    )
}


def load_perturbed_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)
    
    imgs = json_info["images"]
    anns = []
    for img in imgs:
        id = img["id"]
        # find json_info['annotations'] with id
        for ann in json_info["annotations"]:
            if ann["image_id"] == id:
                anns.append(ann)
                break
    assert len(imgs) == len(anns)
    
    ret = []
    for img, ann in zip(imgs, anns):
        image_id = int(ann["image_id"])
        image_file = os.path.join(image_dir, img['file_name'])
        depth_file = os.path.join(image_dir, img["depth_file_name"])
        label_file = os.path.join(image_dir, ann["file_name"].replace('.jpeg', '.png'))
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]

        perturbed_segmentation = img["perturbed_segmentation"]
        tp_mask = img["tp_mask"]
        tn_mask = img["tn_mask"]
        fp_mask = img["fp_mask"]
        fn_mask = img["fn_mask"]
        tp_boundary = img["tp_boundary"]
        tn_boundary = img["tn_boundary"]
        fp_boundary = img["fp_boundary"]
        fn_boundary = img["fn_boundary"]

        ret.append(
            {
                "file_name": image_file,
                "depth_file_name": depth_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "perturbed_segmentation": perturbed_segmentation,
                "tp_mask": tp_mask,
                "tn_mask": tn_mask,
                "fp_mask": fp_mask,
                "fn_mask": fn_mask,
                "tp_boundary": tp_boundary,
                "tn_boundary": tn_boundary,
                "fp_boundary": fp_boundary,
                "fn_boundary": fn_boundary,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_tod_v2_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_perturbed_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_instance_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def _get_tod_v2_panoptic_meta():
    thing_ids = [1]
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    # !TODO: check if this is correct
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = ['object']
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "stuff_classes": [],
        "stuff_colors": [],
        "stuff_dataset_id_to_contiguous_id": {}
    }
    return ret

def register_all_tod_v2_panoptic(root):

    for (prefix,
            (panoptic_root, panoptic_json),
        ) in _PREDEFINED_SPLITS.items():
        prefix_instances = prefix.replace('_panoptic', '').replace('_augmented', '')
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_tod_v2_panoptic(
            prefix,
            _get_tod_v2_panoptic_meta(),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            panoptic_json,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root = 'detectron2_datasets'
register_all_tod_v2_panoptic(_root)
