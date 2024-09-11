#!/bin/bash


# Run main.py with the YAML file
export CUDA_VISIBLE_DEVICES=$1


# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"

# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"

# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"


python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --refiner-model rice
python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --refiner-model cascadepsp

# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OSD --refiner-model rice --visualize
# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OSD --refiner-model cascadepsp --visualize
# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OSD --refiner-model rice --visualize
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OSD --refiner-model cascadepsp --visualize
# python eval/run_eval.py --base-model uoisnet3d --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --refiner-model rice --visualize
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --refiner-model cascadepsp --visualize
# python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
# echo "$yaml_file"
# echo "$1"

# Remove the YAML file from the after directory
# rm "$after_dir/$yaml_file"
done

python eval/run_eval.py --base-model npy --refiner-model cascadepsp_rgbd --test-dataset OSD 
