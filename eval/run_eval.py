
import argparse
import numpy as np
import os

from eval_utils import run_eval



# if __name__ == "__main__":

parser = argparse.ArgumentParser()
# model config   
parser.add_argument("--config-file", 
    default="./configs/quber.yaml", 
    metavar="FILE", help="path to config file")    
parser.add_argument("--gpu", type=str, default="0", help="GPU id")
parser.add_argument("--base-model", 
                    type=str, 
                    default="grounded_sam", 
                    help="Base model for initial segmentation (SAM, Grounded-SAM)")
parser.add_argument("--refiner-model",
                    type=str,
                    default="QuBER",
                    help="Refiner model for instance segmentation (e.g. QuBER, SAM)")
parser.add_argument(
    "--test-dataset",
    type=str,
    default="OSD",
    help="dataset to test on (OSD, OCID)"
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="./datasets/OSD-0.2-depth",
    help="path to the OSD dataset"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="visualize the results"
)
parser.add_argument(
    "--vis-dir",
    type=str,
    default="./vis",
    help="path to the visualization directory"
)
parser.add_argument(
    "--weights-file",
    type=str,
    default="./ckpts/quber.pth",
    help="path to the weights file"
)

# parser.add_argument(
#     "--mask_threshold",
#     type=str,
#     default="0.4",
# )

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

run_eval(args)
