import os
import sys
import cv2
from tqdm import tqdm
import yaml
import argparse
import torch
import torch.nn as nn
from models.late_fusion import create_late_fusion_model

from loader import UOAISSimDataset
from util import *
from monai.losses import *
import glob
import torch.nn.functional as F

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="test", help='config file name')
    parser.add_argument('--gpu_id', default="0", help='gpu_id')
    parser.add_argument('--resume', action='store_true', help='resume training')
    args = parser.parse_args()

    with open('./configs/{}.yaml'.format(args.config)) as f:
        cfg = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''load dataset'''
    train_dataset = UOAISSimDataset(cfg)
    test_dataset = UOAISSimDataset(cfg, train=False)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                                    shuffle=True, num_workers=2)
    val_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                    num_workers=2)

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
    print(model)

    # get this parents directory
    cfg["log_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["log_dir"], args.config)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    print("log_dir: {}".format(cfg["log_dir"]))

    if args.resume:
        ckpts = glob.glob(os.path.join(cfg["log_dir"], "*.pkl"))
        # get the latest checkpoint
        ckpts.sort(key=os.path.getmtime)
        ckpt = ckpts[-1]
        print("resume from {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Loss = getattr(sys.modules[__name__], cfg["loss"])(**cfg["loss_kwargs"])
    best_epoch, best_score = 0, 0
    best_metrics = get_initial_metric(best=True, targets=cfg["targets"], heads=cfg["heads"])

    n_iter = 0
    if args.resume:
        n_iter = int(ckpt.split("/")[-1].split(".")[0].split("_")[-1])
    pbar = tqdm(total=cfg["n_epoch"] * len(train_dataloader))
    # tol_epoch = 5
    for epoch in range(cfg["n_epoch"]):
        model = model.train()
        train_metrics = get_initial_metric(cfg['targets'], cfg['heads'])

        for itr, (data) in enumerate(train_dataloader):
            optimizer.zero_grad()
            for key in data.keys():
                data[key] = data[key].to(device)
            preds = model(data)

            loss_dict = {}
            for head in cfg["heads"]:
                pred = preds[head]
                gt = torch.cat([data[x + '_' + head] for x in cfg["targets"]], dim=1)
                loss_dict[head] = Loss(pred, gt)
                # # true consistency loss (tp + tn)
                # if cfg["true_consistency"] and 'tp' in cfg["targets"] and 'tn' in cfg["targets"]:
                #     if cfg["use_relu_on_consistency"]:
                #         pred_true = F.relu(preds['tp' + "_" + head]) + F.relu(preds['tn' + "_" + head])
                #     else:
                #         pred_true = preds['tp' + "_" + head] + preds['tn' + "_" + head]
                #     gt_true = data['gt_fg_mask']
                #     loss_dict[target + "_true"] = Loss(pred_true, gt_true)
                # # positive consistency loss (tp + fp)
                # if cfg['positive_consistency'] and 'tp' in cfg["targets"] and 'fp' in cfg["targets"]:
                #     if cfg["use_relu_on_consistency"]:
                #         pred_positive = F.relu(preds['tp' + "_" + head]) + F.relu(preds['fp' + "_" + head])
                #     else:
                #         pred_positive = preds['tp' + "_" + head] + preds['fp' + "_" + head]
                #     gt_positive = data['input_fg_mask']
                #     gt_positive = torch.cat([1-gt_positive, gt_positive], dim=1)
                #     loss_dict[target + "_positive"] = Loss(pred_positive, gt_positive)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()

            # calculate metric
            train_metrics = compute_metrics(preds, data, train_metrics, cfg['targets'])
            pbar_dict = {}
            for pred_name in preds.keys():
                pbar_dict[pred_name + "_iou_all"] = train_metrics[pred_name + "_iou_all"][-1]
                pbar_dict[pred_name + "_iou"] = train_metrics[pred_name + "_iou"][-1]
            pbar.set_postfix(pbar_dict)
            pbar.update()
            # visualize the results
            if n_iter % cfg["vis_interval"] == 0:
                model = model.eval()
                for vis_itr, data in enumerate(val_dataloader):
                    if vis_itr > cfg["n_vis"] - 1:
                        break
                    with torch.no_grad():
                        for key in data.keys():
                            data[key] = data[key].to(device)
                        preds = model(data)
                    if not os.path.exists(cfg["log_dir"] + '/uoais-sim'):
                        os.makedirs(cfg["log_dir"] + '/uoais-sim')
                    visualize_inference(data, preds, cfg['targets'], cfg["log_dir"] + '/uoais-sim', 'itr_{}_img_{}'.format(n_iter, vis_itr))

                model = model.train()
            n_iter += 1

        train_metrics = get_average_metrics(train_metrics)    
        for key in train_metrics.keys():
            print("train | {:<20}| {:<7.3f}".format(key, train_metrics[key]))
        model = model.eval()
        val_metrics = get_initial_metric(cfg['targets'], cfg['heads'])
        for val_itr, data in enumerate(val_dataloader):
            with torch.no_grad():
                for key in data.keys():
                    data[key] = data[key].to(device)
                preds = model(data)
                val_metrics = compute_metrics(preds, data, val_metrics, cfg['targets'])
        val_metrics = get_average_metrics(val_metrics)    
        score = 0
        for head in cfg["heads"]:
            score += val_metrics[head + "_iou"]
        score /= len(cfg["targets"]) * len(cfg["heads"])
        if best_score < score : 
            best_score, best_metrics, best_epoch = score, val_metrics, epoch
        torch.save(model.state_dict(), os.path.join(cfg["log_dir"], f"epoch_{epoch}.pth"))
        # else:
            # tol_epoch -= 1
                
        print("epoch: {}| loss: {:.4f}, best_epoch: {}, best_score: {:.4f}".format(epoch, total_loss.item(), best_epoch, best_score))
        for key in val_metrics.keys():
            print("val | {:<20}| {:<7.3f}| best {:<20}| {:<7.3f}".format(key, val_metrics[key], key, best_metrics[key]))
        model = model.train()
        train_metrics = get_initial_metric(cfg['targets'], cfg['heads'])
        # if tol_epoch == 0:
            # print("early stopping")
            # break