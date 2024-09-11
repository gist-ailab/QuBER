import torch
from cgnet import Context_Guided_Network
from lmffnet import LMFFNet
from loader import get_TOD_train_dataloader, get_TOD_test_dataloader
from loss import CELossWeighted
import time 
from tqdm import tqdm

config = {
    "lr": 1e-3,
    "batch_size": 32,
    "num_workers": 2,
    "max_epoch": 100,
    "resume": False,
    "checkpoint": None,
    "TOD_filepath": "/SSDc/Workspaces/seunghyeok_back/mask-refiner/detectron2_datasets/",
    "save_dir": "./results",
    "input": "rgbd",
    # "model": "CGNet"
    "model": "LMFFNet"
}


def train(config):

    if config["input"] == "rgb":
        in_channel = 3
    elif config["input"] == "depth":
        in_channel = 3
    elif config["input"] == "rgbd":
        in_channel = 4
    if config["model"] == "CGNet":
        model = Context_Guided_Network(classes=3, in_channel=in_channel)
    elif config["model"] == "LMFFNet":
        model = LMFFNet(classes=3)
    model.cuda()
    model.train()


    data_loader = get_TOD_train_dataloader(config["TOD_filepath"], config["batch_size"], config["num_workers"])
    total_batches = len(data_loader)

    criterion = CELossWeighted(weighted=True)
    optimizer = torch.optim.RAdam(model.parameters(), lr=config["lr"])
    
    start_epoch = 0
    epoch_loss = []
    for epoch in range(start_epoch, config["max_epoch"]):
        for itr, (rgb_img, depth_img, foreground_labels) in enumerate(tqdm(data_loader)):
            start_time = time.time()
            
            if config["input"] == "rgb":
                img = rgb_img.cuda()
            elif config["input"] == "depth":
                img = depth_img.cuda()
                img = torch.cat([img, img, img], 1)
            elif config["input"] == "rgbd":
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                depth_img = torch.cat([depth_img, depth_img, depth_img], 1)
                img = torch.cat([rgb_img, depth_img], 1)
            foreground_labels = foreground_labels.long().cuda()
            
            logits = model(img)
            loss = criterion(logits, foreground_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            time_taken = time.time() - start_time

            if itr % 100 == 0:
                print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, config["max_epoch"],
                                                                                                itr + 1, total_batches,
                                                                                                config["lr"], loss.item(), time_taken))
            if itr % 1000 == 0:
                model_file_name = config["save_dir"] + '/{}_{}_epoch_{}_itr_{}'.format(config["input"], config["model"], epoch, itr) + '.pth'
                state = {"epoch": epoch, "itr": itr, "model": model.state_dict()}
                torch.save(state, model_file_name)


if __name__ == '__main__':
    
    train(config)