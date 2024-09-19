
```
conda create -n mask-refiner python=3.8
install torch, detectron2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

pip install git+https://github.com/cocodataset/panopticapi.git

pip install timm ninja pyfastnoisesimd monai opencv-python matplotlib imgviz rapidfuzz termcolor imageio setuptools==59.5.0 numpy==1.23.1 pyyaml==5.4.1 open3d easydict

DETECTRON2_DATASETS=

# MSMFormer
pip install -r UnseenObjectsWithMeanShift requirement.txt 
cd UnseenObjectsWithMeanShift/MSMFormer/meanshiftformer/modeling/pixel_decoder/ops && sh make.sh

# RICE
pip install torch_geometric==1.7.2 torch_scatter torch_sparse==0.6.13

# uoais
cd uoais && python setup.py install build

# CascadePSP
pip install segmentation_refinement
```

# Panoptic DeepLab

lecun
cd /SSDc/Workspaces/seunghyeok_back/mask-refiner && conda activate seung
bengio
cd /SSDe/seunghyeok_back/mask-refiner && conda activate seung
ailab3#


CUDA_VISIBLE_DEVICES=1,6 python train_net.py --config-file configs/tod-v2/instance-segmentation/mask-refiner-rgbd-concat-l2-gn-hf-m-b-f-c-o-l3-b8-lr5.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 2


CUDA_VISIBLE_DEVICES=2,6 python train_net.py --config-file configs/uoais-sim/instance-segmentation/mask-refiner-rgbd-concat-l2-gn-hf-o-c-f-b-m-l3-e2-b8.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 2

scp -rv -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i id_container -P 35847 mask-refiner work@127.0.0.1:/home/work/Workspace/seung/


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net.py --config-file configs/tod-v2/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8-freeze-res2.1.yaml --dist-url tcp://127.0.0.1:50159 --num-gpus 4

CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --visualize --config configs/uoais-sim/instance-segmentation/mask-refiner-rgbd-concat-l2-gn-hf-m-b-f-c-o-l3-b8.yaml

CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --visualize --config configs/uoais-sim/instance-segmentation/mask-refiner-rgbd-concat-l2-gn-hf-m-f-c-o-l3-b8.yaml

CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --visualize --config 

python eval/run_eval.py --refiner-model npy --base-model npy


CUDA_VISIBLE_DEVICES=4 python eval/run_eval.py --base-model uoisnet3d --test-dataset OCID --refiner-model cascadepsp


CUDA_VISIBLE_DEVICES=7 python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --refiner-model rice --visualize


CUDA_VISIBLE_DEVICES=0,1 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 2 && CUDA_VISIBLE_DEVICES=0,1 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 2 && CUDA_VISIBLE_DEVICES=0,1 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 2 && CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed777/panoptic-deeplab.yaml --dist-url tcp://127.0.0.1:50158 --num-gpus 1 && CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed7777/panoptic-deeplab.yaml --dist-url tcp://127.0.0.1:50158 --num-gpus 1



CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 2 && CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 2 && CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 2 && CUDA_VISIBLE_DEVICES=6,7 python train_net.py --config-file configs/uoais-sim/instance-segmentation/seed77/panoptic-deeplab.yaml --dist-url tcp://127.0.0.1:50158 --num-gpus 1


CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed77/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed77/panoptic-deeplab.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed777/panoptic-deeplab.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed77/panoptic-deeplab.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed777/panoptic-deeplab.yaml 




CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed7777/panoptic-deeplab.yaml && \
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed7777/panoptic-deeplab.yaml

CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml && \



CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OCID

CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OCID

CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OCID


CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OSD

CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OSD

CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model detic --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-depth-concat-l2-gn-hf-b-fco-l3-b8.yaml  --test-dataset OSD


CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model npy --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 



CUDA_VISIBLE_DEVICES=7 python eval/run_eval.py --base-model grounded-sam --test-dataset OSD --refiner-model maskrefiner --visualize --test-dataset OCID --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 



pip install opencv-python==4.7.0.72


CUDA_VISIBLE_DEVICES=4 python eval/run_eval.py --base-model ucn-zoomin --refiner-model hq-sam --test-dataset OCID --visualize
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --base-model ucn-zoomin --refiner-model hq-sam --test-dataset OSD --visualize

python tools/ours/detection2panoptic_coco_format.py --input_json_file /SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/val_perturbed.json --output_json_file /SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/val_panoptic_perturbed.json --segmentations_folder /SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/val --categories_json_file detectron2_datasets/UOAIS-Sim/annotations/panoptic_armbench_categories.json


CUDA_VISIBLE_DEVICES=0,1 python train_net.py --config-file configs/armbench/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 2


CUDA_VISIBLE_DEVICES=1 python train_net.py --config-file configs/armbench/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml --dist-url tcp://127.0.0.1:50156 --num-gpus 1 --resume --eval-only

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net.py --config-file configs/armbench/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 4 --resume


python train_net.py --config-file configs/armbench/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_please.yaml --dist-url tcp://127.0.0.1:50157 --num-gpus 8 


CUDA_VISIBLE_DEVICES=7 python eval/run_eval.py --base-model msmformer-zoomin --refiner-model hq-sam --visualize --test-dataset OSD 


conda activate mask-refiner
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model ucn-zoomin --refiner-model sam --test-dataset OCID --visualize
CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py --base-model ucn-zoomin --refiner-model hq-sam-pretrained --test-dataset OCID --visualize
CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model ucn-zoomin --refiner-model hq-sam --test-dataset OCID --visualize


CUDA_VISIBLE_DEVICES=3 python eval/run_eval.py --base-model ucn-zoomin --refiner-model sam --test-dataset OCID
CUDA_VISIBLE_DEVICES=4 python eval/run_eval.py --base-model msmformer-zoomin --refiner-model hq-sam-pretrained --test-dataset OCID 
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --base-model msmformer-zoomin --refiner-model hq-sam --test-dataset OCID 



CUDA_VISIBLE_DEVICES=6 python eval/run_eval.py --base-model msmformer-zoomin --refiner-model hq-sam-pretrained --test-dataset WISDOM

CUDA_VISIBLE_DEVICES=7 python eval/run_eval.py --base-model uoaisnet --refiner-model sam --test-dataset WISDOM
CUDA_VISIBLE_DEVICES=4 python eval/run_eval.py --base-model uoaisnet --refiner-model hq-sam-pretrained --test-dataset WISDOM
CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --base-model uoaisnet --refiner-model hq-sam --test-dataset WISDOM


uoaisnet

CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model ucn-zoomin --refiner-model maskrefiner --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml --test-dataset WISDOM


CUDA_VISIBLE_DEVICES=5 python eval/run_eval.py --base-model ucn-zoomin --refiner-model sam --test-dataset OCID 


CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model grounded-sam --test-dataset OSD --refiner-model maskrefiner --test-dataset OCID --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 

CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py --base-model grounded-sam --test-dataset OSD --refiner-model maskrefiner --test-dataset OCID --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 

CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model sam --test-dataset OCID --refiner-model maskrefiner --test-dataset OCID --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 

CUDA_VISIBLE_DEVICES=3 python eval/run_eval.py --base-model sam-depth --test-dataset OCID --refiner-model maskrefiner --test-dataset OCID --config-file /SSDe/seunghyeok_back/mask-refiner/configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgb-concat-l2-gn-hf-b-fco-l3-b8.yaml 


CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --base-model grounded-sam --test-dataset OCID --refiner-model sam 
CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py --base-model grounded-sam --test-dataset OCID --refiner-model hq-sam-pretrained 
CUDA_VISIBLE_DEVICES=3 python eval/run_eval.py --base-model grounded-sam --test-dataset OCID --refiner-model cascadepsp

CUDA_VISIBLE_DEVICES=7 python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --refiner-model hq-sam --visualize