import torch
from cgnet import Context_Guided_Network
from lmffnet import LMFFNet
from loader import get_TOD_train_dataloader, get_TOD_test_dataloader
from loss import CELossWeighted
import time 
from tqdm import tqdm
import glob
import numpy as np
import cv2


config = {
    "lr": 1e-5,
    "batch_size": 1,
    "num_workers": 2,
    "TOD_filepath": "/SSDc/Workspaces/seunghyeok_back/mask-refiner/detectron2_datasets/",
    "save_dir": "results",
    "vis_dir": "vis",
    "model": "LMFFNet",
}

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):  # 预测为正确的像素中确认为正确像素的个数
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):  # 分割正确的像素除以总像素
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def get_iou(data_list, class_num, save_path=None):
    """ 
    Args:
      data_list: a list, its elements [gt, output]
      class_num: the number of label
    """
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    # print(j_list)
    # print(M)
    # print('meanIOU: ' + str(aveJ) + '\n')

    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list


def val(config):

    data_loader = get_TOD_test_dataloader(config["TOD_filepath"], config["batch_size"], config["num_workers"])
    total_batches = len(data_loader)

    if config["model"] == "CGNet":
        checkpoints = glob.glob(config["save_dir"] + "/*CG*.pth")
    elif config["model"] == "LMFFNet":
        checkpoints = glob.glob(config["save_dir"] + "/rgbd_LMFF*.pth")
    print("Found {} checkpoints".format(len(checkpoints)))

    checkpoints = [sorted(checkpoints)][0]
    
    for checkpoint in checkpoints:
        print("Evaluating {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        epoch = checkpoint['epoch']
        if config["model"] == "CGNet":
            model = Context_Guided_Network(classes=1, in_channel=4)
        elif config["model"] == "LMFFNet":
            model = LMFFNet(classes=3)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()

        data_list = []
        for itr, (rgb_img, depth_img, foreground_labels) in enumerate(tqdm(data_loader)):
            start_time = time.time()
            with torch.no_grad():
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                depth_img = torch.cat([depth_img, depth_img, depth_img], 1)
                img = torch.cat([rgb_img, depth_img], 1)
                output = model(img)
            time_taken = time.time() - start_time
            output = output.cpu().data[0].numpy()
            gt = np.asarray(foreground_labels[0].numpy(), dtype=np.uint8)
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            data_list.append([gt.flatten(), output.flatten()])
            if itr > 300:
                break

            if itr % 100 == 0:
                rgb_img = rgb_img.cpu().data[0].numpy()
                rgb_img = rgb_img.transpose(1, 2, 0)
                
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                for i in range(3):
                    rgb_img[...,i] = (rgb_img[...,i] * std[i] + mean[i]) *255
                rgb_img = np.uint8(rgb_img)    
                output = np.expand_dims(output, -1) # (h, w, 1)
                output = np.repeat(output, 3, axis=2)
                # (h, w, 1) -> (h, w, 3) for visualization
                output_vis = np.zeros((output.shape[0], output.shape[1], 3))
                output_vis = np.where(output == 0, [0, 0, 0], output_vis)
                output_vis = np.where(output == 1, [0, 255, 0], output_vis)
                output_vis = np.where(output == 2, [255, 0, 0], output_vis)
                vis = np.hstack([rgb_img, output_vis])

                cv2.imwrite(config["vis_dir"] + "/{}_ep{}_itr{}.jpg".format(config["model"], epoch, itr), vis)

        meanIoU, per_class_iou = get_iou(data_list, 3)
        print("epoch: {}, meanIoU: {}, per_class_iou: {}".format(epoch, meanIoU, per_class_iou))


if __name__ == '__main__':
    
    val(config)