
import argparse
import numpy as np
import os

from un_eval_utils import run_eval



# if __name__ == "__main__":

parser = argparse.ArgumentParser()
# model config   
parser.add_argument("--config-file", 
    default="./configs/uoais-sim/instance-segmentation/mask-refiner-rgbd-concat-l2-gn-hf-m-b-f-c-o-l3-e2-b8.yaml", 
    metavar="FILE", help="path to config file")    
parser.add_argument("--gpu", type=str, default="0", help="GPU id")
parser.add_argument("--base-model", 
                    type=str, 
                    default="uoaisnet", 
                    help="Base model for initial segmentation (uoaisnet, uoisnet3d, ucn)")
parser.add_argument("--refiner-model",
                    type=str,
                    default="maskrefiner",
                    help="Refiner model for instance segmentation (maskrefiner, cascadepsp, rice)")
parser.add_argument(
    "--use-cgnet",
    action="store_true",
    help="Use foreground segmentation model to filter our background instances or not"
)
parser.add_argument(
    "--test-dataset",
    type=str,
    default="OSD",
    help="dataset to test on (OSD, OCID)"
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="./detectron2_datasets/OSD-0.2-depth",
    help="path to the OSD dataset"
)
parser.add_argument(
    "--weights-file",
    type=str,
    default="model_final.pth",
    help="path to the weights file"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize the results"
)
parser.add_argument(
    "--vis_dir",
    type=str,
    default="./vis",
    help="path to the visualization directory"
)
parser.add_argument(
    "--dt_max_distance",
    type=str,
    default="5",
)
parser.add_argument(
    "--mask_threshold",
    type=str,
    default="0.4",
)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

run_eval(args)
