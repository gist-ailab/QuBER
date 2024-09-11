export CUDA_VISIBLE_DEVICES=1


# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config-file configs/tod-v2/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config-file configs/tod-v2/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth
# python eval/run_eval.py --base-model uoaisnet --test-dataset OSD --config-file configs/tod-v2/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth
# python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config-file configs/tod-v2/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth
# python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config-file configs/tod-v2/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth
# python eval/run_eval.py --base-model uoaisnet --test-dataset OCID --config-file configs/tod-v2/instance-segmentation/seed7777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth

# python eval/run_eval.py --base-model msmformer-zoomin --test-dataset OCID --config-file configs/tod-v2/instance-segmentation/seed77/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8.yaml --weights-file model_0089999.pth

# python eval/run_eval.py --base-model detic --test-dataset OCID --refiner-model rice
# python eval/run_eval.py --base-model detic --test-dataset OCID --refiner-model rice

# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OCID --refiner-model cascadepsp
# python eval/run_eval.py --base-model ucn-zoomin --test-dataset OSD --refiner-model cascadepsp

python eval/run_eval.py --base-model detic --test-dataset OSD --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml
python eval/run_eval.py --base-model detic --test-dataset OCID --config-file configs/uoais-sim/instance-segmentation/seed777/mask-refiner-rgbd-concat-l2-gn-hf-b-fco-l3-b8_new.yaml
