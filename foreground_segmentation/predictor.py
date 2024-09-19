import os
import torch
import cv2
import imageio
import numpy as np

try:
    from cgnet import Context_Guided_Network
    from lmffnet import LMFFNet
except:
    from foreground_segmentation.cgnet import Context_Guided_Network
    from foreground_segmentation.lmffnet import LMFFNet

import sys
sys.path.append('/SSDc/Workspaces/seunghyeok_back/mask-refiner')
from eval.preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor

w, h = 320, 240
W, H = 640, 480

class CGNet():

    def __init__(self, weight_path='./foreground_segmentation/rgbd_fg.pth'):
        checkpoint = torch.load(os.path.join(weight_path))
        self.fg_model = Context_Guided_Network(classes=2, in_channel=4)
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
        rgb_img = cv2.resize(rgb_img, (w, h))
        depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        fg_rgb_input = standardize_image(rgb_img)
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = array_to_tensor(depth_img[:,:,0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = self.fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

        return fg_output
    
    

    
class lmffNet():

    def __init__(self, weight_path='./foreground_segmentation/rgbd_lmffnet.pth'):
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

# 
# lmffnet = lmffNet()


# # rgb_path = 'detectron2_datasets/OCID-dataset/ARID10/floor/bottom/curved/seq35/rgb/result_2018-08-23-17-22-33.png'
# # depth_path = 'detectron2_datasets/OCID-dataset/ARID10/floor/bottom/curved/seq35/depth/result_2018-08-23-17-22-33.png'
# # lmffnet.predict(rgb_path, depth_path)
# # rgb_path = '/SSDc/Workspaces/seunghyeok_back/mask-refiner/detectron2_datasets/OCID-dataset/ARID10/floor/top/box/seq03/rgb/result_2018-08-27-15-55-18.png'
# # depth_path = '/SSDc/Workspaces/seunghyeok_back/mask-refiner/detectron2_datasets/OCID-dataset/ARID10/floor/top/box/seq03/depth/result_2018-08-27-15-55-18.png'
# # lmffnet.predict(rgb_path, depth_path)

# import glob
# dataset_path = 'detectron2_datasets/hope_preprocessed'
# rgb_paths = sorted(glob.glob("{}/rgb/*.png".format(dataset_path)))
# depth_paths = sorted(glob.glob("{}/depth/*.png".format(dataset_path)))
# anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))

# for rgb_path, depth_path, anno_path in zip(rgb_paths, depth_paths, anno_paths):
#     lmffnet.predict(rgb_path, depth_path)



# dataset_path = 'detectron2_datasets/OCID-dataset'
# rgb_paths = []
# depth_paths = []
# anno_paths = []
# # load ARID20
# print("... load dataset [ ARID20 ]")
# data_root = dataset_path + "/ARID20"
# f_or_t = ["floor", "table"]
# b_or_t = ["bottom", "top"]
# for dir_1 in f_or_t:
#     for dir_2 in b_or_t:
#         seq_list = sorted(os.listdir(os.path.join(data_root, dir_1, dir_2)))
#         for seq in seq_list:
#             data_dir = os.path.join(data_root, dir_1, dir_2, seq)
#             if not os.path.isdir(data_dir): continue
#             data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
#             for data_name in data_list:
#                 rgb_path = os.path.join(data_root, dir_1, dir_2, seq, "rgb", data_name)
#                 rgb_paths.append(rgb_path)
#                 depth_path = os.path.join(data_root, dir_1, dir_2, seq, "depth", data_name)
#                 depth_paths.append(depth_path)
#                 anno_path = os.path.join(data_root, dir_1, dir_2, seq, "label", data_name)
#                 anno_paths.append(anno_path)
# # load YCB10
# print("... load dataset [ YCB10 ]")
# data_root = dataset_path +  "/YCB10"
# f_or_t = ["floor", "table"]
# b_or_t = ["bottom", "top"]
# c_c_m = ["cuboid", "curved", "mixed"]
# for dir_1 in f_or_t:
#     for dir_2 in b_or_t:
#         for dir_3 in c_c_m:
#             seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
#             for seq in seq_list:
#                 data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
#                 if not os.path.isdir(data_dir): continue
#                 data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
#                 for data_name in data_list:
#                     rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
#                     rgb_paths.append(rgb_path)
#                     depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
#                     depth_paths.append(depth_path)
#                     anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
#                     anno_paths.append(anno_path)
# # load ARID10
# print("... load dataset [ ARID10 ]")
# data_root =  dataset_path + "/ARID10"
# f_or_t = ["floor", "table"]
# b_or_t = ["bottom", "top"]
# c_c_m = ["box", "curved", "fruits", "mixed", "non-fruits"]
# for dir_1 in f_or_t:
#     for dir_2 in b_or_t:
#         for dir_3 in c_c_m:
#             seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
#             for seq in seq_list:
#                 data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
#                 if not os.path.isdir(data_dir): continue
#                 data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
#                 for data_name in data_list:
#                     rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
#                     rgb_paths.append(rgb_path)
#                     depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
#                     depth_paths.append(depth_path)
#                     anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
#                     anno_paths.append(anno_path)
# assert len(rgb_paths) == len(depth_paths)
# assert len(rgb_paths) == len(anno_paths)

# for i in range(len(rgb_paths)):
#     rgb_path = rgb_paths[i]
#     depth_path = depth_paths[i]
#     anno_path = anno_paths[i]
#     print("... inference image [ {} / {} ]".format(i, len(rgb_paths)))
#     lmffnet.predict(rgb_path, depth_path)