cd /SSDe/seunghyeok_back/mask-refiner/explicit_error_estimation && conda activate seung

python train_net.py --gpu 2 --config resnet18_deeplabv3plus_lf_dice_bs16_lr1e-4

python train_net.py --gpu 3 --config resnet34_deeplabv3plus_lf_dice_bs16_lr1e-4

python train_net.py --gpu 4 --config resnet34_deeplabv3plus_lf_dicece_bs16_lr1e-4

python train_net.py --gpu 5 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4

python train_net.py --gpu 6 --config resnet34_pspnet_lf_dice_bs16_lr1e-4

python train_net.py --gpu 7 --config resnet34_deeplabv3plus_lf_dice_bs16_lr1e-4_no_fg_mask_boundary



python train_net.py --gpu 3 --config regnety_002_deeplabv3plus_lf_aug_dicefocal_bs16_lr1e-4
python train_net.py --gpu 4 --config resnet34_pspnet_lf_dicefocal_bs16_lr1e-4
python train_net.py --gpu 5 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4
python train_net.py --gpu 6 --config resnet18_deeplabv3plus_lf_dicefocal_bs16_lr1e-4


python train_net.py --gpu 3 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4_minimal


python train_net.py --gpu 2 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4_minimal

python train_net.py --gpu 6 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4_minimal_tc_pc

python train_net.py --gpu 3 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-3
python train_net.py --gpu 2 --config resnet34_deeplabv3plus_lf_dicefocal_bs16_lr1e-4_softmax

