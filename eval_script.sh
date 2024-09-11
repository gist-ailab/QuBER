#!/bin/bash

 # Define the directories where the YAML files are located
before_dir="configs/uoais-sim/instance-segmentation/seed7777/eval_before"
after_dir="configs/uoais-sim/instance-segmentation/seed7777/after"

# Loop until the before directory is empty
while [ "$(ls -A $before_dir)" ]; do
# Move the first YAML file from the before directory to the after directory
    yaml_file=$(ls $before_dir | head -n 1)
    mv "$before_dir/$yaml_file" "$after_dir/$yaml_file"

    # Run main.py with the YAML file
    export CUDA_VISIBLE_DEVICES=$1
    # python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    # python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/$yaml_file"
    # python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
    # python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    # python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/$yaml_file"
    python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    python eval/run_eval.py --base-model uoisnet3d --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed77/$yaml_file"
    # python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed7777/$yaml_file"
    # echo "$yaml_file"
    # echo "$1"

# Remove the YAML file from the after directory
# rm "$after_dir/$yaml_file"
done