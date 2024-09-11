import os 
import sys
import numpy as np
import cv2
import imageio
import shutil
import glob
import open3d as o3d
import json
from tqdm import tqdm

input_dir = './detectron2_datasets/DoPose'
output_dir = './detectron2_datasets/DoPose'


os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'pcd'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'annotation'), exist_ok=True)


for split in ['test_bin', 'test_table']:
    scene_dirs = sorted(os.listdir(os.path.join(input_dir, split)))
    for scene_dir in tqdm(scene_dirs):
        rgb_dir = os.path.join(input_dir, split, scene_dir, 'rgb')
        scene_camera_path = os.path.join(input_dir, split, scene_dir, 'scene_camera.json')
        with open(scene_camera_path, 'r') as f:
            scene_camera = json.load(f)
        img_names = sorted(os.listdir(rgb_dir))
        for img_name in img_names:
            new_file_name = '{}_{}_{}'.format(split, scene_dir, img_name)
            # copy rgb
            shutil.copyfile(os.path.join(rgb_dir, img_name), os.path.join(output_dir, 'rgb', new_file_name))
            # copy depth
            shutil.copyfile(os.path.join(input_dir, split, scene_dir, 'depth', img_name), os.path.join(output_dir, 'depth', new_file_name))
            mask_visibs = glob.glob(os.path.join(input_dir, split, scene_dir, 'mask_visib', img_name[:-4] + '_*.png'))
            # generate annotation
            for idx, mask_visib in enumerate(mask_visibs):
                mask_visib = imageio.imread(mask_visib)
                mask_visib_ori = mask_visib.copy()
                # fill holes using flood fill, but remain the boundary [0, 255 images]
                mask_visib = cv2.threshold(mask_visib.copy(), 127, 255, cv2.THRESH_BINARY)[1]
                mask_inv = cv2.bitwise_not(mask_visib)
                h, w = mask_visib.shape[:2]
                mask2 = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(mask_visib, mask2, (0, 0), 255)
                mask_visib = cv2.bitwise_not(mask_visib)
                mask_visib = cv2.bitwise_or(mask_visib, mask_visib_ori)
                if idx == 0:
                    annotation = np.zeros_like(mask_visib)
                annotation[mask_visib > 0] = idx + 1
            imageio.imwrite(os.path.join(output_dir, 'annotation', new_file_name), annotation)
            # generate pcd
            rgb_img = imageio.imread(os.path.join(rgb_dir, img_name))
            depth_img = imageio.imread(os.path.join(input_dir, split, scene_dir, 'depth', img_name))
            depth_img = depth_img.astype(np.float32)  # mm to m
            if np.max(depth_img) > 10000: # some files's depth scale is wrong (test_bin 149~)
                depth_img = depth_img / 10
                imageio.imwrite(os.path.join(output_dir, 'depth', new_file_name), np.uint16(depth_img))
            cam_K = np.array(scene_camera[str(int(img_name[:-4]))]['cam_K']).reshape(3, 3)
            # converts 1944x1920 intrinsics to 640x480 intrinsics
            cam_K[0, :] = cam_K[0, :] / 1944 * 640
            cam_K[1, :] = cam_K[1, :] / 1200 * 480


            
            # convert to open3d point cloud
            rgb_img = cv2.resize(rgb_img, (640, 480))
            depth_img = cv2.resize(depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            rgb_img_o3d = o3d.geometry.Image(rgb_img)
            depth_img_o3d = o3d.geometry.Image(depth_img)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(640, 480, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]), project_valid_depth_only=False)
            # print(np.unique(np.asarray(pcd.points)[:, 2]))
            o3d.io.write_point_cloud(os.path.join(output_dir, 'pcd', new_file_name[:-4] + '.pcd'), pcd)


