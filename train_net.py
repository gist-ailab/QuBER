#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from maskrefiner.config import add_mask_refiner_config, add_panoptic_deeplab_config
from maskrefiner import PerturbedPanopticDatasetMapper
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2 import model_zoo



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, use_fast_impl=False))

        # evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_dir=output_folder))
        # if evaluator_type in ["coco_instance_seg"]:
            # evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        #!TODO: check whether test mapper is needed
        mapper = PerturbedPanopticDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PerturbedPanopticDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        freeze_layers = cfg.MODEL.BACKBONE.FREEZE_LAYERS
        weight_path = cfg.MODEL.BACKBONE.WEIGHTS
        print("Loading pretrained weights: ", weight_path)
        if weight_path != "":
        
            pretrained_model = model_zoo.get(weight_path, trained=True)
            pretrained_backbone = pretrained_model
            for name, parameter in model.named_parameters():
                if 'depth' in name:
                    continue
                freeze = False
                for target_name in freeze_layers:
                    if target_name in name:
                        freeze = True
                        break
                if freeze:
                    parameter.requires_grad = False
                    # iterate over all layers in the pretrained weights
                    for pretrained_name, pretrained_parameter in pretrained_backbone.named_parameters():
                        pretrained_name = pretrained_name.replace("bottom_up.", "")
                        if pretrained_name in name:
                            # remove it from params list
                            params = [p for p in params if p["params"] is not parameter]
                            


        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    def freeze(self, cfg):
        # freeze some layers in backbone for training
        # iterate over all layers in the backbone
        freeze_layers = cfg.MODEL.BACKBONE.FREEZE_LAYERS
        weight_path = cfg.MODEL.BACKBONE.WEIGHTS
        print("Loading pretrained weights: ", weight_path)
        if weight_path != "":
            pretrained_model = model_zoo.get(weight_path, trained=True)
            pretrained_backbone = pretrained_model
            for name, parameter in self.model.named_parameters():
                if 'depth' in name:
                    continue
                freeze = False
                for target_name in freeze_layers:
                    if target_name in name:
                        freeze = True
                        break
                if freeze:
                    parameter.requires_grad = False
                    # iterate over all layers in the pretrained weights
                    for pretrained_name, pretrained_parameter in pretrained_backbone.named_parameters():
                        pretrained_name = pretrained_name.replace("bottom_up.", "")
                        if pretrained_name in name:
                            print("Load and freeze pretrained layer: {} from {}".format(name, pretrained_name))
                            try:
                                parameter.data.copy_(pretrained_parameter.data)
                            except:
                                print("Load pretrained layer {} failed".format(pretrained_name))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    add_mask_refiner_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # automatically set output dir
    cfg.OUTPUT_DIR = args.config_file[:-5].replace("configs", "output")
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if not args.eval_only:
        trainer.freeze(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
