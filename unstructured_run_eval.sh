export CUDA_VISIBLE_DEVICES=1

python eval/un_run_eval.py --base-model uoaisnet --test-dataset unstructured_test --refiner-model custom --config-file "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8-10x-R101.yaml" --weights-file model_0629999.pth --visualize