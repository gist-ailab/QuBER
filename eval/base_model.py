import os
import torch
import imageio
import cv2
import sys
import time

import numpy as np
import open3d as o3d
import torch.nn.functional as F
import torchvision

from PIL import Image
from torchvision import transforms as T
from preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import groundingdino.datasets.transforms as TS
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import groundingdino.config.GroundingDINO_SwinT_OGC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR))
from quber.fg_segm import lmffNet


os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


class SAM():

    def __init__(self, dataset, depth_input=False):


        cache_dir = os.path.expanduser("~/.cache/quber")
        sam_checkpoint_path = os.path.join(cache_dir, 'sam_vit_h_4b8939.pth') 
        if not os.path.exists(sam_checkpoint_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Downloading SAM checkpoint to {sam_checkpoint_path}.')
            path = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            torch.hub.download_url_to_file(path, sam_checkpoint_path)
        sam = sam_model_registry["default"](checkpoint=sam_checkpoint_path).cuda()
        self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", min_mask_region_area=300)
        self.predictor = SamPredictor(sam)

        # self.cgnet = CGNet()
        self.H, self.W = 480, 640
        self.lmffnet = lmffNet()
        self.depth_input = depth_input
        self.dataset = dataset

    def predict(self, rgb_path, depth_path):


        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))

        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        if self.dataset == 'OCID':
            _depth_img = imageio.imread(depth_path)
            zero_depth = np.where(_depth_img == 0, True, False)
            zero_depth = zero_depth[:, :, 0] if len(zero_depth.shape) == 3 else zero_depth

        start_time = time.time()
        if self.depth_input:
            masks = self.mask_generator.generate(depth_img)
        else:
            masks = self.mask_generator.generate(rgb_img)
        pred_masks = [x['segmentation'] for x in masks]
        # cluster construct the connected components, and merge the connected components with the same label
        fg_mask = self.lmffnet.predict(rgb_path, depth_path)
        
        filt_masks = []
        for pred_mask in pred_masks:
            pred_mask = np.where(zero_depth, False, pred_mask)
            if np.sum(pred_mask) == 0:
                continue
            if np.sum(np.bitwise_and(pred_mask, fg_mask)) / np.sum(pred_mask) > 0.3:
                filt_masks.append(pred_mask)
        pred_masks = np.asarray(filt_masks)
        time_elapsed = time.time() - start_time
        return pred_masks, fg_mask, time_elapsed # (N, H, W), (H, W)


# modified from https://github.com/FANG-Xiaolin/uncos/blob/main/uncos/groundedsam_wrapper.py
class GroundedSAM:
    def __init__(self, box_thr=0.10, text_thr=0.05):

        config_file = groundingdino.config.GroundingDINO_SwinT_OGC.__file__
        cache_dir = os.path.expanduser("~/.cache/quber")
        grounding_dino_checkpoint_path = os.path.join(cache_dir,
                                                      'groundingdino_swint_ogc.pth')  # change the path of the model
        if not os.path.exists(grounding_dino_checkpoint_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Downloading GroundingDINO checkpoint to {grounding_dino_checkpoint_path}.')
            torch.hub.download_url_to_file(
                'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
                grounding_dino_checkpoint_path)

        sam_checkpoint_path = os.path.join(cache_dir, 'sam_vit_h_4b8939.pth') 
        if not os.path.exists(sam_checkpoint_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Downloading SAM checkpoint to {sam_checkpoint_path}.')
            path = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            torch.hub.download_url_to_file(path, sam_checkpoint_path)

        self.box_threshold = box_thr  # 0.3
        self.text_threshold = text_thr  # 0.05
        self.iou_threshold = 0.5
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # load model
        self.model = self.load_model(config_file, grounding_dino_checkpoint_path)
        self.model = self.model.to(self.device)
        sam = sam_model_registry["default"](checkpoint=sam_checkpoint_path).cuda()
        self.sam_predictor = SamPredictor(sam)
        self.lmffnet = lmffNet()

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(), normalize
        ])
        self.W, self.H = 640, 480

    def predict(self, rgb_path, depth_path):
        # load image
        text_prompt = 'A rigid object.'
        rgb_img = cv2.imread(rgb_path)[:, :, ::-1]
        image_pil = Image.fromarray(np.uint8(rgb_img))
        transform = TS.Compose(
            [
                TS.RandomResize([800], max_size=1333),
                TS.ToTensor(),
                TS.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        pred_masks = np.array([cv2.resize(np.uint8(mask), (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks[:, 0].detach().cpu().numpy()])
        
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
