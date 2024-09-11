# python eval/run_eval.py --base-model npy --test-dataset OSD --refiner-model cascadepsp
# python eval/run_eval.py --base-model npy --test-dataset OSD --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
# python eval/run_eval.py --base-model npy --test-dataset OSD --refiner-model rice

# python eval/run_eval.py --base-model npy --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml"
python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" 


python eval/run_eval.py --base-model npy --test-dataset OCID --refiner-model cascadepsp
python eval/run_eval.py --base-model npy --test-dataset OCID --refiner-model rice


# CUDA_VISIBLE_DEVICES=2 python eval/run_eval.py --base-model uoisnet3d --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# CUDA_VISIBLE_DEVICES=3 python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
# CUDA_VISIBLE_DEVICES=4 python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --config "configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml" --visualize
