import os
import torch
import cv2
import imageio
import numpy as np
from quber.lmffnet import LMFFNet
from eval.preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor
w, h = 320, 240
W, H = 640, 480

    
class lmffNet():

    def __init__(self, weight_path='./ckpts/rgbd_lmffnet.pth'):
        checkpoint = torch.load(os.path.join(weight_path))
        self.fg_model = LMFFNet()
        self.fg_model.load_state_dict(checkpoint['model'])
        self.fg_model.cuda()
        self.fg_model.eval()

    def predict(self, rgb_path, depth_path):

        rgb_img = cv2.imread(rgb_path)
        if 'npy' in depth_path:
            depth_img = np.load(depth_path)
            depth_img = normalize_depth(depth_img, 0.25, 1.5)
        else:
            depth_img = imageio.imread(depth_path)
            depth_img = normalize_depth(depth_img)
        rgb_img = cv2.resize(rgb_img, (W, H))
        # depth_img = normalize_depth(depth_img, min_val=np.unique(depth_img)[1], max_val=np.max(depth_img))
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img, factor=1)

        fg_rgb_input = standardize_image(rgb_img)
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_input = array_to_tensor(depth_img).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_input], 1)
        fg_output = self.fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)

        # output = np.expand_dims(fg_output, -1) # (h, w, 1)
        # output = np.repeat(output, 3, axis=2)
        # # (h, w, 1) -> (h, w, 3) for visualization
        # output_vis = np.zeros((output.shape[0], output.shape[1], 3))
        # output_vis = np.where(output == 0, [0, 0, 0], output_vis)
        # output_vis = np.where(output == 1, [0, 255, 0], output_vis)
        # output_vis = np.where(output == 2, [255, 0, 0], output_vis)
        # vis = np.hstack([rgb_img, depth_img, output_vis])
        # cv2.imwrite('vis/FG/{}.png'.format(os.path.basename(rgb_path)), vis)
        # cv2.imwrite('{}.png'.format(os.path.basename(rgb_path)), vis)
        fg_output = fg_output == 2
        return fg_output
