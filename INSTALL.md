
Install Quber
```
conda create -n quber python=3.8
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

ssh -i AICA_490.pem ubuntu@114.110.134.10

rsync -avz -e "ssh -i ~AICA_490.pem" mix-object-tote ubuntu@114.110.134.10:/data/seunghyeok_back/armbench