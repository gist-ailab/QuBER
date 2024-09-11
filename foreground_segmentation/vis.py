import torch
from foreground_segmentation.cgnet import Context_Guided_Network
from loader import get_TOD_train_dataloader, get_TOD_test_dataloader
from loss import CELossWeighted
import time 
from tqdm import tqdm
import glob
import numpy as np
import cv2
from data_augmentation import standardize_image, array_to_tensor
from tqdm import tqdm
import imageio

config = {
    "lr": 1e-5,
    "batch_size": 1,
    "num_workers": 2,
    "input_dir": "./datasets/OSD/OSD-0.2-depth/image_color",
    "depth_dir": "./datasets/OSD/OSD-0.2-depth/disparity",
    "checkpoint": "./foreground_segmentation/results/depth_model_epoch_5_itr_18000.pth",
    "vis_dir": "./foreground_segmentation/vis",
    "input": "depth"
    
}


def preprocess_depth(depth, depth_min=250, depth_max=1500):

    depth = cv2.inpaint(depth, np.uint8(np.where(depth==0, 1, 0)), 3, cv2.INPAINT_TELEA)
    depth = np.expand_dims(depth, -1)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth = (depth - depth_min) / (depth_max-depth_min) 
    return depth

def visualize(config):



    imgs = glob.glob(config["input_dir"] + "/*.png")
    depths = glob.glob(config["depth_dir"] + "/*.png")
    
    if config["input"] == "rgb":
        in_channel = 3
    elif config["input"] == "depth":
        in_channel = 1
    elif config["input"] == "rgbd":
        in_channel = 4
    model = Context_Guided_Network(classes=2, in_channel=in_channel)
    model.cuda()
    checkpoint = torch.load(config["checkpoint"])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for itr, (rgb, depth) in enumerate(zip(sorted(imgs), sorted(depths))):
        print(itr)
        start_time = time.time()
        
        rgb_img = cv2.imread(rgb)
        rgb_img = standardize_image(rgb_img)
        rgb_img = array_to_tensor(rgb_img).unsqueeze(0)
        
        depth = imageio.imread(depth)
        depth = preprocess_depth(depth)
        depth_img = array_to_tensor(depth*255).unsqueeze(0)
        # depth = np.uint8(np.repeat(depth, 3, -1))
        with torch.no_grad():
            if config["input"] == "rgb":
                img = rgb_img.cuda()
            elif config["input"] == "depth":
                img = depth_img.cuda()
            elif config["input"] == "rgbd":
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                img = torch.cat([rgb_img, depth_img], 1)
            output = model(img)
        time_taken = time.time() - start_time

        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        rgb_img = rgb_img.cpu().data[0].numpy()
        rgb_img = rgb_img.transpose(1, 2, 0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            rgb_img[...,i] = (rgb_img[...,i] * std[i] + mean[i]) *255
        rgb_img = np.uint8(rgb_img)    

        color = np.zeros_like(rgb_img)
        output = output*255
        color[:, :, 1] = output
        vis = cv2.addWeighted(rgb_img, 0.8, color, 0.2, 0)
        cv2.imwrite(config["vis_dir"] + "/osd_{}.jpg".format(itr), vis)


if __name__ == '__main__':
    
    visualize(config)