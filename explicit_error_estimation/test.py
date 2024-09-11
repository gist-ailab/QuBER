import os
import sys
import cv2
from tqdm import tqdm
import yaml
import argparse
import torch
import torch.nn as nn
from models.late_fusion import create_late_fusion_model

from loader import OSDDataset, UOAISSimDataset, OCIDDataset
from util import *
from monai.losses import *
import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="test", help='config file name')
    parser.add_argument('--gpu_id', default="0", help='gpu_id')
    parser.add_argument('--epoch', default=0, help='epoch')
    parser.add_argument('--dataset', default='osd', help='uoais-sim-train, osd')
    args = parser.parse_args()

    with open('./configs/{}.yaml'.format(args.config)) as f:
        cfg = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''load dataset'''
    if args.dataset == 'uoais-sim-train':
        test_dataset = UOAISSimDataset(cfg, train=True)
    elif args.dataset == 'uoais-sim-val':
        test_dataset = UOAISSimDataset(cfg, train=False)
    elif args.dataset == 'osd':
        test_dataset = OSDDataset(cfg)
    elif args.dataset == 'osd-ocid':
        test_dataset = OCIDDataset(cfg)
    test_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} gpu!".format(torch.cuda.device_count()))

    '''load model'''
    model = create_late_fusion_model(
        encoder_name = cfg["encoder_name"],
        encoder_weights = cfg["encoder_weights"],
        encoder_depth = cfg["encoder_depth"],
        encoder_output_stride = cfg["encoder_output_stride"],
        decoder_name = cfg["decoder_name"],
        decoder_dim = cfg["decoder_dim"],
        inputs = cfg["inputs"],
        heads = cfg["heads"],
        targets = cfg["targets"],
        device = device
    )
    model = nn.DataParallel(model)

    # get this parents directory
    cfg["log_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["log_dir"], args.config)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    print("log_dir: {}".format(cfg["log_dir"]))

    ckpt = os.path.join(cfg["log_dir"], "epoch_{}.pth".format(args.epoch))
    print("resume from {}".format(ckpt))
    model.load_state_dict(torch.load(ckpt))

    model = model.eval()
    val_metrics = get_initial_metric(cfg['targets'], cfg['heads'])
    for val_itr, data in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            for key in data.keys():
                data[key] = data[key].to(device)
            preds = model(data)
            output_dir = os.path.join(cfg["log_dir"], args.dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            visualize_inference(data, preds, cfg['targets'], output_dir, 'epoch_{}_img_{}'.format(args.epoch, val_itr))
            val_metrics = compute_metrics(preds, data, val_metrics, cfg['targets'])
    val_metrics = get_average_metrics(val_metrics)    
   
    for key in val_metrics.keys():
        print("test | {:<20}| {:<7.3f}".format(key, val_metrics[key], key,))
