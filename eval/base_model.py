import os
import numpy as np
import imageio
import cv2
import yaml

# uoais
from adet.config import get_cfg
from detectron2.engine import DefaultPredictor
from adet.utils.post_process import detector_postprocess, DefaultPredictor

# import with sys.path.append the current directory
import sys
sys.path.append(os.getcwd())
from uois.src.segmentation import UOISNet3D
from preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor

import open3d as o3d
import torch
from foreground_segmentation.predictor import CGNet, lmffNet
import torch.nn.functional as F

import time
import glob

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def filter_labels_depth(labels, depth, threshold):
    labels_new = labels.clone()
    for i in range(labels.shape[0]):
        label = labels[i]
        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            roi_depth = depth[i, 2][label == mask_id]
            depth_percentage = torch.sum(roi_depth > 0).float() / torch.sum(mask)
            if depth_percentage < threshold:
                labels_new[i][label == mask_id] = 0
    return labels_new


class LoadNpyBaseModel():

    def __init__(self, npy_folder=''):

        self.npy_folder = npy_folder
    
    def predict(self, rgb_path, depth_path):

        npy_path = os.path.join(self.npy_folder, os.path.basename(rgb_path).replace('.png', '.npy'))
        pred_masks = np.load(npy_path)
        pred_masks = [np.where(x>0, True, False) for x in pred_masks]
        pred_masks = np.asarray(pred_masks)
        return pred_masks, None, 0 # (N, H, W), (H, W)
            
class Empty():

    
    def predict(self, rgb_path, depth_path):

        pred_masks = np.asarray([])
        return pred_masks, None, 0 # (N, H, W), (H, W)

class GT():

    def __init__(self, dataset='OSD'):

        self.dataset = dataset
    
    def predict(self, rgb_path, depth_path):

        if self.dataset == "OSD":
            anno_path = rgb_path.replace('image_color', 'annotation')
        elif self.dataset == "OCID":
            anno_path = rgb_path.replace('rgb', 'label')
        BACKGROUND_LABEL = 0
        BG_LABELS = {}
        BG_LABELS["floor"] = [0, 1]
        BG_LABELS["table"] = [0, 1, 2]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (640, 480), interpolation=cv2.INTER_NEAREST)
        if self.dataset == "OCID":
            if "floor" in rgb_path:
                floor_table = "floor"
            elif "table" in rgb_path:
                floor_table = "table"
            for label in BG_LABELS[floor_table]:
                anno[anno == label] = 0         
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        masks = np.array([anno == label for label in labels_anno])
        return masks, None, 0 # (N, H, W), (H, W)

class DynamicArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Detic():

    def __init__(self, repo_path='./Detic', dataset='OCID'):
        
        self.dataset = dataset
        import sys
        sys.path.append(repo_path)

        # Copyright (c) Facebook, Inc. and its affiliates.
        import argparse
        import glob
        import multiprocessing as mp
        import numpy as np
        import os
        import tempfile
        import time
        import warnings
        import cv2
        import tqdm
        import sys
        import mss

        from detectron2.config import get_cfg
        from detectron2.utils.logger import setup_logger

        sys.path.insert(0, repo_path + '/third_party/CenterNet2/')
        from demo_DETIC import setup_cfg
        from centernet.config import add_centernet_config
        from detic.config import add_detic_config

        from detic.predictor import VisualizationDemo
        # args = get_parser().parse_args()
        args = DynamicArgs()


        args.config_file = os.path.join(repo_path, 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
        args.vocabulary = 'custom'
        args.pred_all_class = False
        if dataset == 'OCID':
            args.custom_vocabulary = 'food_box,shampoo,lemon,peach,food_can,potato,flashlight,orange,pear,sponge,ball,bowl,hand_towel,toothpaste,apple,banana,soda_can,cereal_box,coffee_mug,food_bag,keyboard,stapler,tomato,bell_pepper,binder,glue_stick,intant_noodles,kleenex,lime,marker,picher_base,master_chef_can,tuna_finsh_can,mini_soccer_ball,softball,baseball,tennis_ball,racquetball,golf_ball,mug,bleach_cleanser,drill,clamp,chips_can,cracker_box,pudding_box,gelatin_box,wood_block,sugar_box,foam_brick,rubicks_cube,lego_duplo,nine_hole_peg_test,timer'
        elif dataset == 'OSD':
            args.custom_vocabulary = 'box,cereal_box,food_box,block,chips_can,mug,bowl,cookie_can,cylinderic_object,book,cd,drinks,bottle'
        args.confidence_threshold = 0.5
        args.opts = ['MODEL.WEIGHTS', os.path.join(repo_path, 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')]
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        self.demo = VisualizationDemo(cfg, args)
        self.H, self.W = 480, 640


    def predict(self, rgb_path, depth_path):

        from detectron2.data.detection_utils import read_image
        img = read_image(rgb_path, format="BGR")

        start_time = time.time()
        outputs, visualized_output = self.demo.run_on_image(img)
        instances = detector_postprocess(outputs['instances'], self.H, self.W).to('cpu')
        pred_masks = instances.pred_masks.detach().cpu().numpy()
        time_elapsed = time.time() - start_time
        return pred_masks, None, time_elapsed # (N, H, W), (H, W)

class UOAISNet():

    def __init__(self, use_cgnet=True):

        cfg = get_cfg()
        cfg.merge_from_file('uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml')
        cfg.defrost()
        cfg.MODEL.WEIGHTS = os.path.join('uoais', cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

        self.cgnet = CGNet()
        self.H, self.W = 480, 640
    
    def predict(self, rgb_path, depth_path):

        rgb_img = cv2.imread(rgb_path)
        depth_img = imageio.imread(depth_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        uoais_input = np.concatenate([rgb_img, depth_img], -1)   

        start_time = time.time()
        outputs = self.predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], self.H, self.W).to('cpu')
        pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        fg_output = self.cgnet.predict(rgb_path, depth_path)
        filt_masks = []
        for pred_mask in pred_masks:
            if np.sum(pred_mask) == 0:
                continue
            overlap  = np.sum(np.bitwise_and(pred_mask, fg_output)) / np.sum(pred_mask)
            if overlap > 0.5:
                filt_masks.append(pred_mask)
        pred_masks = np.asarray(filt_masks)
        time_elapsed = time.time() - start_time
        return pred_masks, fg_output, time_elapsed # (N, H, W), (H, W)
            

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

class SAM():

    def __init__(self, use_cgnet=True, use_cluster=False):

        sam = sam_model_registry["default"](checkpoint="./sam/sam_vit_h_4b8939.pth").cuda()
        self.use_cluster = use_cluster
        self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", min_mask_region_area=300)
        # else:
        self.predictor = SamPredictor(sam)

        # self.cgnet = CGNet()
        self.H, self.W = 480, 640
        self.lmffnet = lmffNet()

    def predict(self, rgb_path, depth_path):


        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        start_time = time.time()
        masks = self.mask_generator.generate(rgb_img)
        pred_masks = [x['segmentation'] for x in masks]
        # cluster construct the connected components, and merge the connected components with the same label
        
        from scipy import ndimage
        def merge_connected_components(masks):
            """
            Find connected components in binary masks and merge them if connected.
            
            Args:
            masks (np.array): Binary masks of shape [N, H, W]
            
            Returns:
            np.array: Merged masks of shape [N, H, W]
            """
            N, H, W = masks.shape
            merged_masks = np.zeros_like(masks)
            
            for n in range(N):
                # Find connected components
                labeled, num_features = ndimage.label(masks[n])
                
                # If there's only one component or no components, no need to merge
                if num_features <= 1:
                    merged_masks[n] = masks[n]
                    continue
                
                # Create a new mask for merged components
                merged_mask = np.zeros((H, W), dtype=bool)
                
                # Merge all connected components
                for i in range(1, num_features + 1):
                    merged_mask |= labeled == i
                
                merged_masks[n] = merged_mask
            
            return merged_masks
        
        # if self.use_cluster:
            # pred_masks = merge_connected_components(np.asarray(pred_masks))
        # fg_mask = self.cgnet.predict(rgb_path, depth_path)
        fg_mask = self.lmffnet.predict(rgb_path, depth_path)
        
        filt_masks = []
        for pred_mask in pred_masks:
            if np.sum(np.bitwise_and(pred_mask, fg_mask)) / np.sum(pred_mask) > 0.3:
                filt_masks.append(pred_mask)
        pred_masks = np.asarray(filt_masks)
        time_elapsed = time.time() - start_time
        return pred_masks, fg_mask, time_elapsed # (N, H, W), (H, W)

import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# segment anything
from segment_anything import build_sam, SamPredictor

import torchvision.transforms as TS
import matplotlib.pyplot as plt

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# refac from https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/automatic_label_ram_demo.py
# code from https://github.com/FANG-Xiaolin/uncos/blob/main/uncos/groundedsam_wrapper.py
class GroundedSAM:
    def __init__(self, box_thr=0.2, text_thr=0.05):
        import groundingdino.config.GroundingDINO_SwinT_OGC
        config_file = groundingdino.config.GroundingDINO_SwinT_OGC.__file__
        cache_dir = os.path.expanduser("~/.cache/uncos")
        grounding_dino_checkpoint_path = os.path.join(cache_dir,
                                                      'groundingdino_swint_ogc.pth')  # change the path of the model
        if not os.path.exists(grounding_dino_checkpoint_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Downloading GroundingDINO checkpoint to {grounding_dino_checkpoint_path}.')
            torch.hub.download_url_to_file(
                'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
                grounding_dino_checkpoint_path)
        device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        sam = sam_model_registry["default"](checkpoint="./sam/sam_vit_h_4b8939.pth").cuda()
        self.box_threshold = box_thr  # 0.3
        self.text_threshold = text_thr  # 0.05
        self.iou_threshold = 0.5
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # load model
        self.model = self.load_model(config_file, grounding_dino_checkpoint_path)
        self.model = self.model.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        self.lmffnet = lmffNet()

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), normalize
        ])
        self.W, self.H = 640, 480

    def predict(self, rgb_path, depth_path):
        # load image
        text_prompt = 'A rigid object.'
        # text_prompt = 'box,cereal box,food box,block,chips_can,mug,bowl,cookie_can,cylinderic_object,book,cd,drinks,Bottle'
        text_prompt = 'Box. Cereal box. Food box. Block. Chips can. Mug. Bowl. Cookie can. Cylinderic object. Book. CD. Drinks. Bottle.'
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))[:, :, ::-1]
        image_pil = Image.fromarray(np.uint8(rgb_img))
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_normalized, _ = transform(image_pil, None)  # 3, h, w

        image_rgb_255 = np.array(image_pil)
        tags = text_prompt
        # run grounding dino model
        start_time = time.time()
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            image_normalized, tags
        )

        self.sam_predictor.set_image(image_rgb_255)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        if len(boxes_filt) == 0:
            return np.array([]), None, 0

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_rgb_255.shape[:2]).to(
            self.device)
        masks, iou_predictions, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        pred_masks = np.array([mask.astype(bool) for mask in masks[:, 0].detach().cpu().numpy()])
        
        fg_mask = self.lmffnet.predict(rgb_path, depth_path)
        filt_masks = []
        for pred_mask in pred_masks:
            if np.sum(np.bitwise_and(pred_mask, fg_mask)) / np.sum(pred_mask) > 0.3:
                filt_masks.append(pred_mask)
        filt_masks = np.asarray(filt_masks)
        time_elapsed = time.time() - start_time
        return filt_masks, None, time_elapsed

    def get_grounding_output(self, image, caption):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image[np.newaxis], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        # logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        # logits_filt.shape[0]

        # get phrase
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        # print(load_res)
        _ = model.eval()
        return model




class UOISNet():
    def __init__(self, repo_path='./uois', dataset='OCID'):
        uoisnet3d_cfg_filename = repo_path + '/uoisnet3d.yaml'
        dsn_filename = repo_path + '/checkpoint_dir/DepthSeedingNetwork_3D_TOD_checkpoint.pth' 
        rrn_filename = repo_path + '/checkpoint_dir/RRN_OID_checkpoint.pth' 
        
        with open(uoisnet3d_cfg_filename, 'r') as f:
            uoisnet_3d_config = yaml.load(f, Loader=yaml.FullLoader)

        self.uois_net = UOISNet3D(uoisnet_3d_config['uois_config'], 
                        dsn_filename,
                        uoisnet_3d_config['dsn_config'],
                        rrn_filename,
                        uoisnet_3d_config['rrn_config'])
        
        self.dataset = dataset

    def predict(self, rgb_path, depth_path):

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        rgb_img = standardize_image(rgb_img)

        if self.dataset == 'OSD' or self.dataset == "unstructured_test":
            pcd_path = depth_path.replace('disparity', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        elif self.dataset in ['OCID', 'HOPE', 'DoPose']:
            pcd_path = depth_path.replace('depth', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        H, W = rgb_img.shape[:2]
        xyz_img = np.asarray(pcd.points).reshape((H, W, 3))
        xyz_img[np.isnan(xyz_img)] = 0
        # multiply -1 to y axis
        xyz_img[:, :, 1] *= -1


        batch = {
            'rgb' : array_to_tensor(np.expand_dims(rgb_img, 0)),
            'xyz' : array_to_tensor(np.expand_dims(xyz_img, 0)),
        }

        ### Compute segmentation masks ###
        start_time = time.time()
        fg_masks, center_offsets, initial_masks, seg_masks = self.uois_net.run_on_batch(batch)
        time_elapsed = time.time() - start_time
        seg_masks = seg_masks.detach().cpu().numpy()[0] # (H, W)
        pred_masks = []
        for i in np.unique(seg_masks):
            if i == 0:
                continue
            pred_mask = seg_masks == i
            pred_masks.append(pred_mask)
        pred_masks = np.asarray(pred_masks)
        fg_masks = fg_masks.detach().cpu()[0] # (H, W)
        fg_mask = fg_masks == 2 # 1: table, 2: object
        return pred_masks, fg_mask, time_elapsed
    
class UCN():
    def __init__(self, repo_path='./UnseenObjectClustering', dataset='OCID', zoom_in=False): 

        network_name = 'seg_resnet34_8s_embedding'
        pretrained = './UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth'
        cfg_file = './UnseenObjectClustering/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml'
        pretrained_crop = './UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth'

        self.dataset = dataset
        self.zoom_in = zoom_in
        sys.path.append('UnseenObjectClustering/lib')

        from UnseenObjectClustering.lib.fcn.config import cfg, cfg_from_file
        from UnseenObjectClustering.lib import networks

        cfg_from_file(cfg_file)
        self.cfg = cfg
        self.cfg.MODE = 'TEST'
        self._pixel_mean = torch.tensor(self.cfg.PIXEL_MEANS/255.0).cuda()
        network_data = torch.load(pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(pretrained))
        self.network = networks.__dict__[network_name](2, cfg.TRAIN.NUM_UNITS, network_data).cuda()
        self.network = torch.nn.DataParallel(self.network, device_ids=[cfg.gpu_id]).cuda()
        self.network.eval()

        if self.zoom_in: 
            network_data_crop = torch.load(pretrained_crop)
            self.network_crop = networks.__dict__[network_name](2, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()
            self.network_crop = torch.nn.DataParallel(self.network_crop, device_ids=[cfg.gpu_id]).cuda()
            self.network_crop.eval()

    def predict(self, rgb_path, depth_path):

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        im_tensor = torch.from_numpy(rgb_img).cuda() / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).float().unsqueeze(0)
        if self.dataset == 'OSD' or self.dataset == "unstructured_test":
            pcd_path = depth_path.replace('disparity', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        elif self.dataset in ['OCID', 'HOPE', 'DoPose']:
            pcd_path = depth_path.replace('depth', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        xyz_img = np.asarray(pcd.points).reshape((480, 640, 3)) 
        xyz_img[np.isnan(xyz_img)] = 0

        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1).float().unsqueeze(0)
        start_time = time.time()
        features = self.network(image_blob, None, depth_blob)
        out_label, selected_pixels = self.clustering_features(features, num_seeds=100)
        if self.dataset == 'OSD':
            out_label = filter_labels_depth(out_label, depth_blob, 0.8)
        elif self.dataset in ['OCID', 'HOPE', 'DoPose']:
            out_label = filter_labels_depth(out_label, depth_blob, 0.5)
        prediction = out_label.squeeze().detach().cpu().numpy()
        time_elapsed = time.time() - start_time

        if self.zoom_in:
            rgb_crop, out_label_crop, rois, depth_crop = self.crop_rois(image_blob, out_label.clone(), depth_blob)
            if rgb_crop.shape[0] > 0:
                features_crop = self.network_crop(rgb_crop, out_label_crop, depth_crop)
                labels_crop, selected_pixels_crop = self.clustering_features(features_crop)
                out_label_refined, labels_crop = self.match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
                prediction = out_label_refined.squeeze().detach().cpu().numpy()

        pred_masks = []
        for i in np.unique(prediction):
            if i == 0:
                continue
            pred_mask = prediction == i
            pred_masks.append(pred_mask)
        pred_masks = np.asarray(pred_masks)
        return pred_masks, None, time_elapsed

    def clustering_features(self, features, num_seeds=100):
        metric = self.cfg.TRAIN.EMBEDDING_METRIC
        height = features.shape[2]
        width = features.shape[3]
        out_label = torch.zeros((features.shape[0], height, width))

        # mean shift clustering
        kappa = 20
        selected_pixels = []
        for j in range(features.shape[0]):
            X = features[j].view(features.shape[1], -1)
            X = torch.transpose(X, 0, 1)
            cluster_labels, selected_indices = self.mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric=metric)
            out_label[j] = cluster_labels.view(height, width)
            selected_pixels.append(selected_indices)
        return out_label, selected_pixels

    def mean_shift_smart_init(self, X, kappa, num_seeds=100, max_iters=10, metric='cosine'):
        """ Runs mean shift with carefully selected seeds
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param dist_threshold: parameter for the von Mises-Fisher distribution
            @param num_seeds: number of seeds used for mean shift clustering
            @return: a [n] array of cluster labels
        """

        n, d = X.shape
        seeds, selected_indices = self.select_smart_seeds(X, num_seeds, return_selected_indices=True, metric=metric)
        seed_cluster_labels, updated_seeds = self.mean_shift_with_seeds(X, seeds, kappa, max_iters=max_iters, metric=metric)

        # Get distances to updated seeds
        if metric == 'euclidean':
            distances = X.unsqueeze(1) - updated_seeds.unsqueeze(0)  # a are points, b are seeds
            distances = torch.norm(distances, dim=2)
        elif metric == 'cosine':
            distances = 0.5 * (1 - torch.mm(X, updated_seeds.t())) # Shape: [n x num_seeds]

        # Get clusters by assigning point to closest seed
        closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
        cluster_labels = seed_cluster_labels[closest_seed_indices]

        # assign zero to the largest cluster
        num = len(torch.unique(seed_cluster_labels))
        count = torch.zeros(num, dtype=torch.long)
        for i in range(num):
            count[i] = (cluster_labels == i).sum()
        label_max = torch.argmax(count)
        if label_max != 0:
            index1 = cluster_labels == 0
            index2 = cluster_labels == label_max
            cluster_labels[index1] = label_max
            cluster_labels[index2] = 0

        return cluster_labels, selected_indices
    
    def select_smart_seeds(self, X, num_seeds, return_selected_indices=False, init_seeds=None, num_init_seeds=None, metric='cosine'):
        """ Selects seeds that are as far away as possible
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param num_seeds: number of seeds to pick
            @param init_seeds: a [num_seeds x d] vector of initial seeds
            @param num_init_seeds: the number of seeds already chosen.
                                the first num_init_seeds rows of init_seeds have been chosen already
            @return: a [num_seeds x d] matrix of seeds
                    a [n x num_seeds] matrix of distances
        """
        n, d = X.shape
        selected_indices = -1 * torch.ones(num_seeds, dtype=torch.long)

        # Initialize seeds matrix
        if init_seeds is None:
            seeds = torch.empty((num_seeds, d), device=X.device)
            num_chosen_seeds = 0
        else:
            seeds = init_seeds
            num_chosen_seeds = num_init_seeds

        # Keep track of distances
        distances = torch.empty((n, num_seeds), device=X.device)

        if num_chosen_seeds == 0:  # Select first seed if need to
            selected_seed_index = np.random.randint(0, n)
            selected_indices[0] = selected_seed_index
            selected_seed = X[selected_seed_index, :]
            seeds[0, :] = selected_seed
            if metric == 'euclidean':
                distances[:, 0] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
            elif metric == 'cosine':
                distances[:, 0] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])  
            num_chosen_seeds += 1
        else:  # Calculate distance to each already chosen seed
            for i in range(num_chosen_seeds):
                if metric == 'euclidean':
                    distances[:, i] = torch.norm(X - seeds[i:i+1, :], dim=1)
                elif metric == 'cosine':
                    distances[:, i] = 0.5 * (1 - torch.mm(X, seeds[i:i+1, :].t())[:, 0])

        # Select rest of seeds
        for i in range(num_chosen_seeds, num_seeds):
            # Find the point that has the furthest distance from the nearest seed
            distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: [n]
            selected_seed_index = torch.argmax(distance_to_nearest_seed)
            selected_indices[i] = selected_seed_index
            selected_seed = torch.index_select(X, 0, selected_seed_index)[0, :]
            seeds[i, :] = selected_seed

            # Calculate distance to this selected seed
            if metric == 'euclidean':
                distances[:, i] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
            elif metric == 'cosine':
                distances[:, i] = 0.5 * (1 - torch.mm(X, selected_seed.unsqueeze(1))[:,0])

        return_tuple = (seeds,)
        if return_selected_indices:
            return_tuple += (selected_indices,)
        return return_tuple
    
    def connected_components(self, Z, epsilon, metric='cosine'):
        """
            For the connected components, we simply perform a nearest neighbor search in order:
                for each point, find the points that are up to epsilon away (in cosine distance)
                these points are labeled in the same cluster.
            @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints
            @return: a [n] torch.LongTensor of cluster labels
        """
        n, d = Z.shape

        K = 0
        cluster_labels = torch.ones(n, dtype=torch.long) * -1
        for i in range(n):
            if cluster_labels[i] == -1:

                if metric == 'euclidean':
                    distances = Z.unsqueeze(1) - Z[i:i + 1].unsqueeze(0)  # a are points, b are seeds
                    distances = torch.norm(distances, dim=2)
                elif metric == 'cosine':
                    distances = 0.5 * (1 - torch.mm(Z, Z[i:i+1].t()))
                component_seeds = distances[:, 0] <= epsilon

                # If at least one component already has a label, then use the mode of the label
                if torch.unique(cluster_labels[component_seeds]).shape[0] > 1:
                    temp = cluster_labels[component_seeds].numpy()
                    temp = temp[temp != -1]
                    label = torch.tensor(self.get_label_mode(temp))
                else:
                    label = torch.tensor(K)
                    K += 1  # Increment number of clusters

                cluster_labels[component_seeds] = label

        return cluster_labels

    def seed_hill_climbing_ball(self, X, Z, kappa, max_iters=10, metric='cosine'):
        """ Runs mean shift hill climbing algorithm on the seeds.
            The seeds climb the distribution given by the KDE of X
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
            @param dist_threshold: parameter for the ball kernel
        """
        n, d = X.shape
        m = Z.shape[0]

        for _iter in range(max_iters):

            # Create a new object for Z
            new_Z = Z.clone()

            W = self.ball_kernel(Z, X, kappa, metric=metric)

            # use this allocated weight to compute the new center
            new_Z = torch.mm(W, X)  # Shape: [n x d]

            # Normalize the update
            if metric == 'euclidean':
                summed_weights = W.sum(dim=1)
                summed_weights = summed_weights.unsqueeze(1)
                summed_weights = torch.clamp(summed_weights, min=1.0)
                Z = new_Z / summed_weights
            elif metric == 'cosine':
                Z = F.normalize(new_Z, p=2, dim=1)

        return Z

    def mean_shift_with_seeds(self, X, Z, kappa, max_iters=10, metric='cosine'):
        """ Runs mean-shift
            @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
            @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
            @param dist_threshold: parameter for the von Mises-Fisher distribution
        """

        Z = self.seed_hill_climbing_ball(X, Z, kappa, max_iters=max_iters, metric=metric)

        # Connected components
        cluster_labels = self.connected_components(Z, 2 * self.cfg.TRAIN.EMBEDDING_ALPHA, metric=metric)  # Set epsilon = 0.1 = 2*alpha

        return cluster_labels, Z
    
    def ball_kernel(self, Z, X, kappa, metric='cosine'):
        """ Computes pairwise ball kernel (without normalizing constant)
            (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)
            @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
            @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points
            @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
                    without normalizing constant
        """
        if metric == 'euclidean':
            distance = Z.unsqueeze(1) - X.unsqueeze(0)
            distance = torch.norm(distance, dim=2)
            kernel = torch.exp(-kappa * torch.pow(distance, 2))
        elif metric == 'cosine':
            kernel = torch.exp(kappa * torch.mm(Z, X.t()))
        return kernel

    def get_label_mode(self, array):
        """ Computes the mode of elements in an array.
            Ties don't matter. Ties are broken by the smallest value (np.argmax defaults)
            @param array: a numpy array
        """
        labels, counts = np.unique(array, return_counts=True)
        mode = labels[np.argmax(counts)].item()
        return mode


    def crop_rois(self, rgb, initial_masks, depth):

        N, H, W = initial_masks.shape
        crop_size = self.cfg.TRAIN.SYN_CROP_SIZE
        padding_percentage = 0.25

        mask_ids = torch.unique(initial_masks[0])
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = mask_ids.shape[0]
        rgb_crops = torch.zeros((num, 3, crop_size, crop_size)).cuda()
        rois = torch.zeros((num, 4)).cuda()
        mask_crops = torch.zeros((num, crop_size, crop_size)).cuda()
        if depth is not None:
            depth_crops = torch.zeros((num, 3, crop_size, crop_size)).cuda()
        else:
            depth_crops = None

        for index, mask_id in enumerate(mask_ids):
            mask = (initial_masks[0] == mask_id).float() # Shape: [H x W]
            a = torch.nonzero(mask)
            x_min, y_min, x_max, y_max = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
            x_padding = int(torch.round((x_max - x_min).float() * padding_percentage).item())
            y_padding = int(torch.round((y_max - y_min).float() * padding_percentage).item())

            # pad and be careful of boundaries
            x_min = max(x_min - x_padding, 0)
            x_max = min(x_max + x_padding, W-1)
            y_min = max(y_min - y_padding, 0)
            y_max = min(y_max + y_padding, H-1)
            rois[index, 0] = x_min
            rois[index, 1] = y_min
            rois[index, 2] = x_max
            rois[index, 3] = y_max

            # crop
            rgb_crop = rgb[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]
            if depth is not None:
                depth_crop = depth[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]

            # resize
            new_size = (crop_size, crop_size)
            rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
            rgb_crops[index] = rgb_crop
            mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
            mask_crops[index] = mask_crop
            if depth is not None:
                depth_crop = F.upsample_bilinear(depth_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
                depth_crops[index] = depth_crop

        return rgb_crops, mask_crops, rois, depth_crops


    # labels_crop is the clustering labels from the local patch
    def match_label_crop(self, initial_masks, labels_crop, out_label_crop, rois, depth_crop):
        num = labels_crop.shape[0]
        for i in range(num):
            mask_ids = torch.unique(labels_crop[i])
            for index, mask_id in enumerate(mask_ids):
                mask = (labels_crop[i] == mask_id).float()
                overlap = mask * out_label_crop[i]
                percentage = torch.sum(overlap) / torch.sum(mask)
                if percentage < 0.5:
                    labels_crop[i][labels_crop[i] == mask_id] = -1

        # sort the local labels
        sorted_ids = []
        for i in range(num):
            if depth_crop is not None:
                if torch.sum(labels_crop[i] > -1) > 0:
                    roi_depth = depth_crop[i, 2][labels_crop[i] > -1]
                else:
                    roi_depth = depth_crop[i, 2]
                avg_depth = torch.mean(roi_depth[roi_depth > 0])
                sorted_ids.append((i, avg_depth))
            else:
                x_min = rois[i, 0]
                y_min = rois[i, 1]
                x_max = rois[i, 2]
                y_max = rois[i, 3]
                orig_H = y_max - y_min + 1
                orig_W = x_max - x_min + 1
                roi_size = orig_H * orig_W
                sorted_ids.append((i, roi_size))

        sorted_ids = sorted(sorted_ids, key=lambda x : x[1], reverse=True)
        sorted_ids = [x[0] for x in sorted_ids]

        # combine the local labels
        refined_masks = torch.zeros_like(initial_masks).float()
        count = 0
        for index in sorted_ids:

            mask_ids = torch.unique(labels_crop[index])
            if mask_ids[0] == -1:
                mask_ids = mask_ids[1:]

            # mapping
            label_crop = torch.zeros_like(labels_crop[index])
            for mask_id in mask_ids:
                count += 1
                label_crop[labels_crop[index] == mask_id] = count

            # resize back to original size
            x_min = int(rois[index, 0].item())
            y_min = int(rois[index, 1].item())
            x_max = int(rois[index, 2].item())
            y_max = int(rois[index, 3].item())
            orig_H = int(y_max - y_min + 1)
            orig_W = int(x_max - x_min + 1)
            mask = label_crop.unsqueeze(0).unsqueeze(0).float()
            resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0, 0]

            # Set refined mask
            h_idx, w_idx = torch.nonzero(resized_mask).t()
            refined_masks[0, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = resized_mask[h_idx, w_idx].cpu()

        return refined_masks, labels_crop

class MSMFormer():

    def __init__(self, repo_path='./uois', dataset='OCID', zoom_in=False):
        

        cfg_file_MSMFormer = os.path.join('UnseenObjectsWithMeanShift/MSMFormer/configs/tabletop_pretrained.yaml')
        weight_path_MSMFormer = os.path.join("UnseenObjectsWithMeanShift/data/checkpoints/norm_model_0069999.pth")


        cfg_file_MSMFormer_crop = os.path.join("UnseenObjectsWithMeanShift/MSMFormer/configs/crop_tabletop_pretrained.yaml")
        weight_path_MSMFormer_crop = os.path.join("UnseenObjectsWithMeanShift/data/checkpoints/crop_dec9_model_final.pth")

        self.predictor, self.cfg = self.get_general_predictor(cfg_file_MSMFormer, weight_path_MSMFormer, 'RGBD_ADD')
        self.predictor_crop, self.cfg_crop = self.get_general_predictor(cfg_file_MSMFormer_crop, weight_path_MSMFormer_crop, 'RGBD_ADD')
        self.dataset = dataset
        self.zoom_in = zoom_in
        self._pixel_mean = torch.tensor(np.array([[[102.9801, 115.9465, 122.7717]]]) / 255.0).float()

    def get_general_predictor(self, cfg_file, weight_path, input_image="RGBD_ADD"):
        
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.config import get_cfg
        sys.path.append('UnseenObjectsWithMeanShift/MSMFormer')
        from meanshiftformer.config import add_meanshiftformer_config
        from tabletop_config import add_tabletop_config
        sys.path.append('UnseenObjectsWithMeanShift/lib')
        from fcn.test_utils import Network_RGBD

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_meanshiftformer_config(cfg)
        cfg_file = cfg_file
        cfg.merge_from_file(cfg_file)
        add_tabletop_config(cfg)
        cfg.SOLVER.IMS_PER_BATCH = 1  #

        cfg.INPUT.INPUT_IMAGE = input_image
        if input_image == "RGBD_ADD":
            cfg.MODEL.USE_DEPTH = True
        else:
            cfg.MODEL.USE_DEPTH = False
        # arguments frequently tuned
        cfg.TEST.DETECTIONS_PER_IMAGE = 20
        weight_path = weight_path
        cfg.MODEL.WEIGHTS = weight_path
        predictor = Network_RGBD(cfg)
        return predictor, cfg

    def get_confident_instances(self, outputs, topk=False, score=0.7, num_class=2, low_threshold=0.4):
        """
        Extract objects with high prediction scores.
        """
        instances = outputs["instances"]
        if topk:
            # we need to remove background predictions
            # keep only object class
            if num_class >= 2:
                instances = instances[instances.pred_classes == 1]
                confident_instances = instances[instances.scores > low_threshold]
                return confident_instances
            else:
                return instances
        confident_instances = instances[instances.scores > score]
        return confident_instances

    def combine_masks_with_NMS(self, instances):
        """
        Combine several bit masks [N, H, W] into a mask [H,W],
        e.g. 8*480*640 tensor becomes a numpy array of 480*640.
        [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
        """
        mask = instances.get('pred_masks').to('cpu').numpy()
        scores = instances.get('scores').to('cpu').numpy()

        # non-maximum suppression
        keep = self.nms(mask, scores, thresh=0.7).astype(int)
        mask = mask[keep]
        scores = scores[keep]

        num, h, w = mask.shape
        bin_mask = np.zeros((h, w))
        score_mask = np.zeros((h, w))
        num_instance = len(mask)
        bbox = np.zeros((num_instance, 5), dtype=np.float32)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask, score_mask, bbox

        for m, object_label in zip(mask, range(2, 2 + num_instance)):
            label_pos = np.nonzero(m)
            bin_mask[label_pos] = object_label
            score_mask[label_pos] = int(scores[object_label - 2] * 100)

            # bounding box
            y1 = np.min(label_pos[0])
            y2 = np.max(label_pos[0])
            x1 = np.min(label_pos[1])
            x2 = np.max(label_pos[1])
            bbox[object_label - 2, :] = [x1, y1, x2, y2, scores[object_label - 2]]

        return bin_mask, score_mask, bbox
    
    def nms(self, masks, scores, thresh):

        n = masks.shape[0]
        inters = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                inters[i, j] = np.sum(np.multiply(masks[i], masks[j]))
                inters[j, i] = inters[i, j]

        areas = np.diag(inters)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            inter = inters[i, order[1:]]
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        index = np.argsort(areas[keep]).astype(np.int32)
        return np.array(keep)[index]

    def crop_rois(self, rgb, initial_masks, depth):
        N, H, W = initial_masks.shape
        crop_size = 224
        padding_percentage = 0.25

        mask_ids = torch.unique(initial_masks[0])
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = mask_ids.shape[0]
        rgb_crops = torch.zeros((num, 3, crop_size, crop_size)).cuda()
        rois = torch.zeros((num, 4)).cuda()
        mask_crops = torch.zeros((num, crop_size, crop_size)).cuda()
        if depth is not None:
            depth_crops = torch.zeros((num, 3, crop_size, crop_size)).cuda()
        else:
            depth_crops = None

        for index, mask_id in enumerate(mask_ids):
            mask = (initial_masks[0] == mask_id).float() # Shape: [H x W]
            a = torch.nonzero(mask)
            x_min, y_min, x_max, y_max = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
            x_padding = int(torch.round((x_max - x_min).float() * padding_percentage).item())
            y_padding = int(torch.round((y_max - y_min).float() * padding_percentage).item())

            # pad and be careful of boundaries
            x_min = max(x_min - x_padding, 0)
            x_max = min(x_max + x_padding, W-1)
            y_min = max(y_min - y_padding, 0)
            y_max = min(y_max + y_padding, H-1)
            rois[index, 0] = x_min
            rois[index, 1] = y_min
            rois[index, 2] = x_max
            rois[index, 3] = y_max

            # crop
            rgb_crop = rgb[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]
            if depth is not None:
                depth_crop = depth[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]

            # resize
            new_size = (crop_size, crop_size)
            rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
            rgb_crops[index] = rgb_crop
            mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
            mask_crops[index] = mask_crop
            if depth is not None:
                depth_crop = F.upsample_bilinear(depth_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
                depth_crops[index] = depth_crop

        return rgb_crops, mask_crops, rois, depth_crops
    
    def get_result_from_network(self, cfg, image, depth, label, predictor, topk=False, confident_score=0.7, low_threshold=0.4, vis_crop=False):
        height = image.shape[-2]  # image: 3XHXW, tensor
        width = image.shape[-1]
        image = torch.squeeze(image, dim=0)
        depth = torch.squeeze(depth, dim=0)

        sample = {"image": image, "height": height, "width": width, "depth": depth}
        outputs = predictor(sample)
        confident_instances = self.get_confident_instances(outputs, topk=topk, score=confident_score,
                                                    num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                    low_threshold=low_threshold)
        binary_mask, score_mask, bbox = self.combine_masks_with_NMS(confident_instances)
        return binary_mask

    def combine_masks_with_NMS(self, instances):
        """
        Combine several bit masks [N, H, W] into a mask [H,W],
        e.g. 8*480*640 tensor becomes a numpy array of 480*640.
        [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
        """
        mask = instances.get('pred_masks').to('cpu').numpy()
        scores = instances.get('scores').to('cpu').numpy()

        # non-maximum suppression
        keep = self.nms(mask, scores, thresh=0.7).astype(int)
        mask = mask[keep]
        scores = scores[keep]

        num, h, w = mask.shape
        bin_mask = np.zeros((h, w))
        score_mask = np.zeros((h, w))
        num_instance = len(mask)
        bbox = np.zeros((num_instance, 5), dtype=np.float32)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask, score_mask, bbox

        for m, object_label in zip(mask, range(2, 2 + num_instance)):
            label_pos = np.nonzero(m)
            bin_mask[label_pos] = object_label
            score_mask[label_pos] = int(scores[object_label - 2] * 100)

            # bounding box
            y1 = np.min(label_pos[0])
            y2 = np.max(label_pos[0])
            x1 = np.min(label_pos[1])
            x2 = np.max(label_pos[1])
            bbox[object_label - 2, :] = [x1, y1, x2, y2, scores[object_label - 2]]

        return bin_mask, score_mask, bbox

    def match_label_crop(self, initial_masks, labels_crop, out_label_crop, rois, depth_crop):
        num = labels_crop.shape[0]
        for i in range(num):
            mask_ids = torch.unique(labels_crop[i])
            for index, mask_id in enumerate(mask_ids):
                mask = (labels_crop[i] == mask_id).float()
                overlap = mask * out_label_crop[i]
                percentage = torch.sum(overlap) / torch.sum(mask)
                if percentage < 0.5:
                    labels_crop[i][labels_crop[i] == mask_id] = -1

        # sort the local labels
        sorted_ids = []
        for i in range(num):
            if depth_crop is not None:
                if torch.sum(labels_crop[i] > -1) > 0:
                    roi_depth = depth_crop[i, 2][labels_crop[i] > -1]
                else:
                    roi_depth = depth_crop[i, 2]
                avg_depth = torch.mean(roi_depth[roi_depth > 0])
                sorted_ids.append((i, avg_depth))
            else:
                x_min = rois[i, 0]
                y_min = rois[i, 1]
                x_max = rois[i, 2]
                y_max = rois[i, 3]
                orig_H = y_max - y_min + 1
                orig_W = x_max - x_min + 1
                roi_size = orig_H * orig_W
                sorted_ids.append((i, roi_size))

        sorted_ids = sorted(sorted_ids, key=lambda x : x[1], reverse=True)
        sorted_ids = [x[0] for x in sorted_ids]

        # combine the local labels
        refined_masks = torch.zeros_like(initial_masks).float()
        count = 0
        for index in sorted_ids:

            mask_ids = torch.unique(labels_crop[index])
            if mask_ids[0] == -1:
                mask_ids = mask_ids[1:]

            # mapping
            label_crop = torch.zeros_like(labels_crop[index])
            for mask_id in mask_ids:
                count += 1
                label_crop[labels_crop[index] == mask_id] = count

            # resize back to original size
            x_min = int(rois[index, 0].item())
            y_min = int(rois[index, 1].item())
            x_max = int(rois[index, 2].item())
            y_max = int(rois[index, 3].item())
            orig_H = int(y_max - y_min + 1)
            orig_W = int(x_max - x_min + 1)
            mask = label_crop.unsqueeze(0).unsqueeze(0).float()
            resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0, 0]

            # Set refined mask
            h_idx, w_idx = torch.nonzero(resized_mask).t()
            refined_masks[0, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = resized_mask[h_idx, w_idx]#.cpu()  # in mean shift mask Transformer, disable cpu()

        return refined_masks, labels_crop

    def predict(self, rgb_path, depth_path):

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        im_tensor = torch.from_numpy(rgb_img) / 255.0
        im_tensor_bgr = im_tensor.clone()
        im_tensor_bgr = im_tensor_bgr.permute(2, 0, 1)
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).float()
        if self.dataset == 'OSD':
            pcd_path = depth_path.replace('disparity', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        elif self.dataset in ['OCID', 'HOPE', 'DoPose']:
            pcd_path = depth_path.replace('depth', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        xyz_img = np.asarray(pcd.points).reshape((480, 640, 3))
        xyz_img[np.isnan(xyz_img)] = 0

        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1).float()

        sample = {
                  'image': image_blob,
                  'depth': depth_blob,
                  'height': 480,
                  'width': 640,
                  }
        start_time = time.time()
        outputs = self.predictor(sample)
        confident_instances = self.get_confident_instances(outputs, topk=False, score=0.7,
                                                  num_class=self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=0.4)
        binary_mask, score_mask, bbox = self.combine_masks_with_NMS(confident_instances)
        time_elapsed = time.time() - start_time

        if self.zoom_in:
            out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()
            if len(depth_blob.shape) == 3:
                depth_blob = torch.unsqueeze(depth_blob, dim=0)
            if len(image_blob.shape) == 3:
                image_blob = torch.unsqueeze(image_blob, dim=0)
            if self.dataset == 'OSD':
                out_label = filter_labels_depth(out_label, depth_blob, 0.8)
            elif self.dataset in ['OCID', 'HOPE', 'DoPose']:
                out_label = filter_labels_depth(out_label, depth_blob, 0.5)
            start_time = time.time()
            # zoom in refinement
            rgb_crop, out_label_crop, rois, depth_crop = self.crop_rois(image_blob, out_label.clone(), depth_blob)
            if rgb_crop.shape[0] > 0:
                labels_crop = torch.zeros((rgb_crop.shape[0], rgb_crop.shape[-2], rgb_crop.shape[-1]))#.cuda()
                for i in range(rgb_crop.shape[0]):
                    if depth_crop is None:
                        binary_mask_crop = self.get_result_from_network(self.cfg, rgb_crop[i], None, out_label_crop[i],
                                                                self.predictor_crop,
                                                                topk=False, confident_score=0.7,
                                                                low_threshold=0.4)
                    else:
                        binary_mask_crop = self.get_result_from_network(self.cfg, rgb_crop[i], depth_crop[i], out_label_crop[i], self.predictor_crop,
                                                        topk=False, confident_score=0.7, low_threshold=0.4)
                    labels_crop[i] = torch.from_numpy(binary_mask_crop)
                binary_mask, labels_crop = self.match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
                binary_mask = binary_mask.squeeze(dim=0).cpu().numpy()
            time_elapsed += time.time() - start_time

        pred_masks = []
        for i in np.unique(binary_mask):
            if i == 0:
                continue
            pred_mask = binary_mask == i
            pred_masks.append(pred_mask)
        pred_masks = np.array(pred_masks)

        return pred_masks, None, time_elapsed