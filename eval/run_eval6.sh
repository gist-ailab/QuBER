python eval/run_eval.py --base-model npy --refiner-model npy --test-dataset OCID --dt_max_distance 5 --mask_threshold 0.6
python eval/run_eval.py --base-model npy --refiner-model npy --test-dataset OCID --dt_max_distance 10 --mask_threshold 0.6
python eval/run_eval.py --base-model npy --refiner-model npy --test-dataset OCID --dt_max_distance 15 --mask_threshold 0.6
python eval/run_eval.py --base-model npy --refiner-model npy --test-dataset OCID --dt_max_distance 20 --mask_threshold 0.6
python eval/run_eval.py --base-model npy --refiner-model npy --test-dataset OCID --dt_max_distance 25 --mask_threshold 0.6

#* ucn zoom in 만 visualize 하면 됨
#* dataset 별로 inference time, 이미지 읽어오고 저장하는거 제외
#* offset map 만드는거 + segfix inference time 각각 따로 측정
#* np.std
