import os
import numpy as np
import imageio
import yaml

# uoais
from adet.config import get_cfg
from detectron2.engine import DefaultPredictor
from adet.utils.post_process import detector_postprocess, DefaultPredictor

# import with sys.path.append the current directory
import sys
sys.path.append(os.getcwd())
from preprocess_utils import standardize_image, inpaint_depth, normalize_depth, array_to_tensor, compute_xyz

# import parent directory of this file
# get parent of current file path
sys.path.append('/SSDe/seunghyeok_back/mask-refiner/ext_modules')
import rice.src.data_augmentation as data_augmentation
import rice.src.graph_construction as gc
import rice.src.graph_networks as gn
import rice.src.merge_split_networks as msn
import rice.src.delete_network as delnet
import rice.src.sample_tree_cem as stc
import rice.src.network_config as nc

import open3d as o3d
from maskrefiner.predictor import MaskRefinerPredictor
import segmentation_refinement as refine
import torch
import time

from foreground_segmentation.predictor import CGNet, lmffNet
from base_model import UOISNet
from maskrefiner.modeling.mask_refiner.post_processing import get_panoptic_segmentation
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.data import MetadataCatalog
import copy
import cv2

# from CascadePSP.models.cascadepsp_rgbd import process_high_res_im_depth
# from CascadePSP.models.psp.pspnet import PSPNet_UOAIS
from torchvision import transforms
import detectron2


W = 640 
H = 480

os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# filter labels on zero depths
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




import typing
from collections import defaultdict

import tabulate
from torch import nn

def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(model: nn.Module, max_depth: int = 10) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table. It looks like this:

    ::

        | name                            | #elements or shape   |
        |:--------------------------------|:---------------------|
        | model                           | 37.9M                |
        |  backbone                       |  31.5M               |
        |   backbone.fpn_lateral3         |   0.1M               |
        |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
        |    backbone.fpn_lateral3.bias   |    (256,)            |
        |   backbone.fpn_output3          |   0.6M               |
        |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output3.bias    |    (256,)            |
        |   backbone.fpn_lateral4         |   0.3M               |
        |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
        |    backbone.fpn_lateral4.bias   |    (256,)            |
        |   backbone.fpn_output4          |   0.6M               |
        |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output4.bias    |    (256,)            |
        |   backbone.fpn_lateral5         |   0.5M               |
        |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
        |    backbone.fpn_lateral5.bias   |    (256,)            |
        |   backbone.fpn_output5          |   0.6M               |
        |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output5.bias    |    (256,)            |
        |   backbone.top_block            |   5.3M               |
        |    backbone.top_block.p6        |    4.7M              |
        |    backbone.top_block.p7        |    0.6M              |
        |   backbone.bottom_up            |   23.5M              |
        |    backbone.bottom_up.stem      |    9.4K              |
        |    backbone.bottom_up.res2      |    0.2M              |
        |    backbone.bottom_up.res3      |    1.2M              |
        |    backbone.bottom_up.res4      |    7.1M              |
        |    backbone.bottom_up.res5      |    14.9M             |
        |    ......                       |    .....             |

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        # if x > 1e8:
        #     return "{:.5f}G".format(x / 1e9)
        if x > 1e5:
            return "{:.5f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.5f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab


class LoadNpyRefinerModel():

    def __init__(self, npy_folder='', dataset='OSD'):

        self.npy_folder = npy_folder
        self.dataset = dataset

    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        npy_path = os.path.join(self.npy_folder, os.path.basename(rgb_path).replace('.png', '.npy'))
        refined_masks = np.load(npy_path)
        refined_masks = [np.where(x>0, True, False) for x in refined_masks]
        if self.dataset == 'OCID':
            depth_img = imageio.imread(depth_path)
            zero_depth = np.where(depth_img == 0, True, False)
            for refined_mask in refined_masks:
                refined_mask[zero_depth] = False
        refined_masks = np.asarray(refined_masks)
        return refined_masks, None, 0, None
            

class MaskRefiner():

    def __init__(self, config_file, weights_file, dataset='OSD'):

        self.refiner_predictor = MaskRefinerPredictor(config_file, weights_file= weights_file)
        self.lmffnet = lmffNet()
        self.dataset = dataset
        # print the number of parameters


    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        if self.dataset == 'armbench':
            rgb_img = cv2.imread(rgb_path)
            h, w = detectron2.data.transforms.ResizeShortestEdge.get_output_shape(rgb_img.shape[0], rgb_img.shape[1], 800, 1333)
            rgb_img = cv2.resize(rgb_img, (w, h))
            if initial_masks.dtype == np.bool:
                initial_masks = np.uint8(initial_masks) * 255
            initial_masks = np.array([cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) for mask in initial_masks])
            

            start_time = time.time()
            output = self.refiner_predictor.predict(rgb_img, None, initial_masks)[0]
            
            if "instances" not in output.keys():
                refined_masks = []
            else:
                refined_instances = output['instances'].to('cpu')
                refined_masks = refined_instances.pred_masks.detach().cpu().numpy()
            fg_mask = None
            time_elapsed = time.time() - start_time

        else:
            rgb_img = cv2.imread(rgb_path)
            if 'npy' in depth_path:
                depth_img = np.load(depth_path)
            else:
                depth_img = imageio.imread(depth_path)
            rgb_img = cv2.resize(rgb_img, (W, H))
            zero_depth = np.where(depth_img == 0)
            if 'npy' in depth_path:
                depth_img = normalize_depth(depth_img, 0.25, 1.5)
            else:
                depth_img = normalize_depth(depth_img)
            depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_img = inpaint_depth(depth_img)

            if initial_masks.dtype == np.bool:
                initial_masks = np.uint8(initial_masks) * 255
            

            start_time = time.time()
            output = self.refiner_predictor.predict(rgb_img, depth_img, initial_masks)[0]
            if "instances" not in output.keys():
                refined_masks = []
            else:
                refined_instances = output['instances'].to('cpu')
                refined_masks = refined_instances.pred_masks.detach().cpu().numpy()
            time_elapsed = time.time() - start_time
            fg_mask = self.lmffnet.predict(rgb_path, depth_path)
            filt_masks = []
            for refined_mask in refined_masks:
                if np.sum(np.bitwise_and(refined_mask, fg_mask)) / np.sum(refined_mask) > 0.3:
                    filt_masks.append(refined_mask)
            time_elapsed = time.time() - start_time
            if self.dataset == 'OCID':
                # The methods using the xyz images (e.g RICE, UOIS, UCN, MSMFormer) automatically filter out the zero-depth pixels.
                # DoPose dataset's labels are 0 for zero-depth pixels.
                # Thus, we need to filter out the zero-depth pixels for these two datasets.
                filt_masks2 = []
                for refined_mask in filt_masks:
                    refined_mask[zero_depth] = False
                    filt_masks2.append(refined_mask)
                filt_masks = filt_masks2
                refined_masks = np.asarray(filt_masks)


            # events = prof.events()
            # forward_flops = sum([int(evt.flops) for evt in events]) 
            # time_elapsed = forward_flops / 1e9
            # print(time_elapsed)
            # time.sleep(1000)

        return refined_masks, output, time_elapsed, fg_mask # [N, H, W], bool


class CascadePSP():

    def __init__(self, L=900, fast=False, dataset='OSD', depth=False):

        self.refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'
        self.L = L
        self.fast = fast
        self.dataset = dataset

        self.depth = depth

        # trained on uoais-sim dataset
        if self.depth:
            model_path = '/ailab_mat/personal/maeng_jemo/unstructured/CascadePSP/weights/20240329_depth_pretrained_2024-03-29_07:32:55/model_40000'
            self.refiner.model = PSPNet_UOAIS(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50_uoais', pretrained=False)
            model_dict = torch.load(model_path, map_location={'cuda:0': 'cuda:0'})
            new_dict = {}
            for k, v in model_dict.items():
                name = k[7:] # Remove module. from dataparallel
                new_dict[name] = v
            self.refiner.model.load_state_dict(new_dict)

        # else:
            # model_path = '/ailab_mat/personal/maeng_jemo/unstructured/CascadePSP/weights/20240401_rgbonly_pretrained_2024-04-01_02:18:22/model_40000'
            # model_dict = torch.load(model_path, map_location={'cuda:0': 'cuda:0'})
            # new_dict = {}
            # for k, v in model_dict.items():
            #     print(k)
            #     name = k[7:] # Remove module. from dataparallel
            #     new_dict[name] = v
            # self.refiner.model.load_state_dict(new_dict)
        self.refiner.model.eval().to('cuda:0')

        

    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))

        if self.depth:
            depth_img = imageio.imread(depth_path)
            depth_img = normalize_depth(depth_img)
            depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_img = inpaint_depth(depth_img)
            depth_img = depth_img.astype(np.float32) / 255


        if self.dataset == 'OCID':
            _depth_img = imageio.imread(depth_path)
            zero_depth = np.where(_depth_img == 0, True, False)
            zero_depth = zero_depth[:, :, 0] if len(zero_depth.shape) == 3 else zero_depth
        if initial_masks.dtype == np.bool:
            initial_masks = np.uint8(initial_masks) * 255
        
        refined_masks = []
        start_time = time.time()
        # with torch.profiler.profile(
        # activities=[
        #     torch.profiler.ProfilerActivity.CPU,
        #     torch.profiler.ProfilerActivity.CUDA,
        # ],
        # with_flops=True) as prof:
        for initial_mask in initial_masks:
            if self.depth:
                with torch.no_grad():
                    im_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                    ])
                    seg_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.5],
                            std=[0.5]
                        ),
                    ])
                    image = im_transform(rgb_img).unsqueeze(0).to('cuda:0')
                    mask = seg_transform((initial_mask>127).astype(np.uint8)*255).unsqueeze(0).to('cuda:0')
                    if len(mask.shape) < 4:
                        mask = mask.unsqueeze(0)
                    depth = torch.Tensor(np.expand_dims(depth_img, 0)).to('cuda:0')
                    # [1, 480, 640, 3] to [1, 1, 480, 640]
                    depth = depth.permute(0, 3, 1, 2)
                    depth = depth[:, 0:1, :, :]
                    output = process_high_res_im_depth(self.refiner.model, image, depth, mask, self.L)
                    refined_mask = (output[0,0].cpu().numpy()*255).astype('uint8')

            else:
                refined_mask = self.refiner.refine(rgb_img, initial_mask, L=self.L, fast=self.fast)
            if self.dataset == 'OCID':
                refined_mask = np.where(zero_depth, False, refined_mask)
            refined_masks.append(refined_mask)
        # events = prof.events()
        # forward_flops = sum([int(evt.flops) for evt in events]) 
        # time_elapsed = forward_flops / 1e9
        # print(time_elapsed)
        time_elapsed = time.time() - start_time
        refined_masks = np.asarray(refined_masks, dtype=bool)
        return refined_masks, None, time_elapsed, None # [N, H, W], bool


class RICE():

    def __init__(self, repo_path='./ext_modules/rice', base_model='uoisnet3d', dataset='OSD'):
        # SplitNet
        splitnet_config = nc.get_splitnet_config(repo_path + '/configs/splitnet.yaml')
        sn_wrapper = msn.SplitNetWrapper(splitnet_config)
        sn_filename = repo_path + '/checkpoint_dir/SplitNetWrapper_checkpoint.pth'  
        sn_wrapper.load(sn_filename)
        # sn_wrapper.load('/SSDe/seunghyeok_back/mask-refiner/rice/joint_split_delete_training_uoais_sim/SplitNetWrapper_iter200000_checkpoint.pth')

        # MergeNet (uses SplitNet under the hood)
        merge_net_config = splitnet_config.copy()
        merge_net_config['splitnet_model'] = sn_wrapper.model
        mn_wrapper = msn.MergeBySplitWrapper(merge_net_config)

        # DeleteNet
        deletenet_config = nc.get_deletenet_config(repo_path + '/configs/deletenet.yaml')
        dn_wrapper = delnet.DeleteNetWrapper(deletenet_config)
        delnet_filename = repo_path + '/checkpoint_dir/DeleteNetWrapper_checkpoint.pth'  
        dn_wrapper.load(delnet_filename)
        # dn_wrapper.load('/SSDe/seunghyeok_back/mask-refiner/rice/joint_split_delete_training_uoais_sim/DeleteNetWrapper_iter200000_checkpoint.pth')

        # SGS-Net
        sgsnet_config = nc.get_sgsnet_config(repo_path + '/configs/sgsnet.yaml')
        
        sgsnet_wrapper = gn.SGSNetWrapper(sgsnet_config)
        sgsnet_filename = repo_path + '/checkpoint_dir/SGSNetWrapper_checkpoint.pth'  
        sgsnet_wrapper.load(sgsnet_filename)
        # sgsnet_wrapper.load('/SSDe/seunghyeok_back/mask-refiner/rice/sgsnet_training_uoais_sim_200000/SGSNetWrapper_iter5000_checkpoint.pth')

        rn50_fpn = gc.get_resnet50_fpn_model(pretrained=True)

        # Load RICE
        with open(repo_path + '/configs/rice.yaml', 'r') as f:
            rice_config = yaml.load(f, Loader=yaml.FullLoader)
        sample_operator_networks = {
            'mergenet_wrapper' : mn_wrapper,
            'splitnet_wrapper' : sn_wrapper,
            'deletenet_wrapper' : dn_wrapper,
        }
        self.rice = stc.SampleTreeCEMWrapper(
            rn50_fpn,
            sample_operator_networks,
            sgsnet_wrapper,
            rice_config,
        )


        self.base_model = base_model
        if base_model != 'uoisnet3d':
            self.uoisnet = UOISNet(dataset=dataset)
            print('Use UOISNet to get fg_mask')

        self.dataset = dataset
        # print(parameter_count_table(self.uoisnet.uois_net.dsn.model))
        # print(parameter_count_table(self.uoisnet.uois_net.rrn.model))
        # exit()

    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgb_img = standardize_image(rgb_img)

        #!TODO: support OCID dataset
        if self.dataset == 'OSD' or self.dataset == "unstructured_test":
            pcd_path = depth_path.replace('disparity', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        elif self.dataset in ['OCID', 'DoPose']:
            pcd_path = depth_path.replace('depth', 'pcd').replace('.png', '.pcd')
            if not os.path.exists(pcd_path):
                print('No pcd file found at {0}'.format(pcd_path))
                raise FileNotFoundError
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        H, W = rgb_img.shape[:2]
        xyz_img = np.asarray(pcd.points).reshape((H, W, 3))
        xyz_img[np.isnan(xyz_img)] = 0

        # depth_img = imageio.imread(depth_path) / 1000.0
        # xyz_img = compute_xyz(depth_img, camera_params={"fx": 577.3, "fy": 579.4, "x_offset": 320, "y_offset": 240, "img_width": 640, "img_height": 480})

        # [N, H, W] to [1, H, W] with 0, 2, 3, 4, 5, 6
        seg_masks = np.zeros((H, W), dtype=np.uint8)
        for i, mask in enumerate(initial_masks):
            seg_masks[mask] = i + 2

        if self.base_model != 'uoisnet3d':
            # if not using uoisnet3d, use uoisnet to get fg_mask
            _, fg_mask, _ = self.uoisnet.predict(rgb_path, depth_path)

        batch = {
            'rgb' : array_to_tensor(np.expand_dims(rgb_img, 0)),
            'xyz' : array_to_tensor(np.expand_dims(xyz_img, 0)),
            'seg_masks': torch.Tensor(np.expand_dims(seg_masks, 0)), # [1, H, W], 0,2,3,4,5,6
            'fg_mask': torch.Tensor(np.expand_dims(fg_mask, 0)).to(torch.bool) # [1, H, W], F, T
        }
        start_time = time.time()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     with_flops=True) as prof:
        #     sample_tree = self.rice.run_on_batch(batch, verbose=False)  # this is where the magic happens!
        # events = prof.events()
        # forward_flops = sum([int(evt.flops) for evt in events]) 
        # time_elapsed = forward_flops / 1e9 
        # print(time_elapsed)
        # time.sleep(1000)
        sample_tree = self.rice.run_on_batch(batch, verbose=False)  # this is where the magic happens!
        time_elapsed = time.time() - start_time
        scores = np.array([g.sgs_net_score for g in sample_tree.all_graphs()])
        best_node = sample_tree.all_nodes()[np.argmax(scores)]
        refined_masks = best_node.graph.orig_masks.cpu().numpy()
        refined_masks = np.array(refined_masks, dtype=bool)[1:] # first one is background

        return refined_masks, None, time_elapsed, fg_mask # [N, H, W], bool
    

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

class SAMRefiner():

    def __init__(self, prompt_type='mask', dataset='OSD', hq=False, pretrained=False):

        # else:
        if hq:
            from segment_anything_hq import sam_model_registry, SamPredictor
            if pretrained:
                self.sam = sam_model_registry['vit_h'](checkpoint='./sam/sam_hq_vit_h.pth')
            else:
                self.sam = sam_model_registry['vit_h'](checkpoint='/SSDe/seunghyeok_back/mask-refiner/sam-hq/train/work_dirs/hq_sam_h_uoais_sim_new/sam_hq_epoch_0.pth')
            self.sam = self.sam.cuda()
            self.sam_predictor = SamPredictor(self.sam)
        else:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam = sam_model_registry["default"](checkpoint="./sam/sam_vit_h_4b8939.pth").cuda()
            self.sam_predictor = SamPredictor(self.sam)

        # self.cgnet = CGNet()
        self.H, self.W = 480, 640
        self.dataset = dataset
        self.prompt_type = prompt_type

    def _process_box(self, box, shape, original_size=None, box_extension=0):
        if box_extension == 0:  # no extension
            extension_y, extension_x = 0, 0
        elif box_extension >= 1:  # extension by a fixed factor
            extension_y, extension_x = box_extension, box_extension
        else:  # extension by fraction of the box len
            len_y, len_x = box[2] - box[0], box[3] - box[1]
            extension_y, extension_x = box_extension * len_y, box_extension * len_x

        box = np.array([
            max(box[1] - extension_x, 0), max(box[0] - extension_y, 0),
            min(box[3] + extension_x, shape[1]), min(box[2] + extension_y, shape[0]),
        ])

        if original_size is not None:
            trafo = ResizeLongestSide(max(original_size))
            box = trafo.apply_boxes(box[None], (256, 256)).squeeze()
        return box

    def _compute_box_from_mask(self, mask, original_size=None, box_extension=0):
        coords = np.where(mask == 1)
        min_y, min_x = coords[0].min(), coords[1].min()
        max_y, max_x = coords[0].max(), coords[1].max()
        box = np.array([min_y, min_x, max_y + 1, max_x + 1])
        return self._process_box(box, mask.shape, original_size=original_size, box_extension=box_extension)

    def _compute_logits_from_mask(self, mask, eps=1e-3):

        def inv_sigmoid(x):
            return np.log(x / (1 - x))

        # Resize to have the longest side 256.
        resized_mask, new_height, new_width = self.resize_mask(mask)

        # Add padding to make it square.
        square_mask = self.pad_mask(resized_mask, new_height, new_width, False)

        # Expand SAM mask's dimensions to 1xHxW (1x256x256).
        logits = np.zeros(square_mask.shape, dtype="float32")
        logits[square_mask == 1] = 1 - eps
        logits[square_mask == 0] = eps
        logits = inv_sigmoid(logits)

    
        logits = logits[None]
        return logits

    def resize_mask(self, ref_mask: np.ndarray, longest_side: int = 256):
        """
        Resize an image to have its longest side equal to the specified value.

        Args:
            ref_mask (np.ndarray): The image to be resized.
            longest_side (int, optional): The length of the longest side after resizing. Default is 256.

        Returns:
            tuple[np.ndarray, int, int]: The resized image and its new height and width.
        """
        height, width = ref_mask.shape[:2]
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))

        return (
            cv2.resize(
                ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            ),
            new_height,
            new_width,
        )

    def pad_mask(
        self,
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
        ) -> np.ndarray:
        """
        Add padding to an image to make it square.

        Args:
            ref_mask (np.ndarray): The image to be padded.
            new_height (int): The height of the image after resizing.
            new_width (int): The width of the image after resizing.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

        Returns:
            np.ndarray: The padded image.
        """
        pad_height = 256 - new_height
        pad_width = 256 - new_width
        if pad_all_sides:
            padding = (
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            )
        else:
            padding = ((0, pad_height), (0, pad_width))

        # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
        return np.pad(ref_mask, padding, mode="constant")

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

    def combine_masks_with_NMS(self, mask, scores):
        """
        Combine several bit masks [N, H, W] into a mask [H,W],
        e.g. 8*480*640 tensor becomes a numpy array of 480*640.
        [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
        """

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

    def predict(self, rgb_path, depth_path, initial_masks, fg_mask):

        #!TODO: support OCID dataset

        if self.dataset == 'OCID':
            _depth_img = imageio.imread(depth_path)
            zero_depth = np.where(_depth_img == 0, True, False)
            zero_depth = zero_depth[:, :, 0] if len(zero_depth.shape) == 3 else zero_depth

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        self.sam_predictor.reset_image()
        start_time = time.time()
        self.sam_predictor.set_image(rgb_img, image_format="BGR")
        
        
        # https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L375-L388
        h, w = self.sam.prompt_encoder.mask_input_size
        pred_masks = []
        scores = []
        for initial_mask in initial_masks:
            initial_mask = initial_mask.astype(np.uint8)
            box = self._compute_box_from_mask(initial_mask, box_extension=0.0)
            logits = self._compute_logits_from_mask(initial_mask)
            masks, iou_predictions, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box = box,
                mask_input=logits,
                multimask_output=True,
                )
            for mask, score in zip(masks, iou_predictions):
                mask, _ = remove_small_regions(mask, area_thresh=300, mode="holes")
                if np.sum(mask) == 0:
                    continue
                
                pred_masks.append(mask)
                scores.append(score)
        if len(pred_masks) != 0:
            masks, _, _ = self.combine_masks_with_NMS(np.array(pred_masks), np.array(scores))
            pred_masks = []
            for id in np.unique(masks)[1:]:
                pred_masks.append(masks == id)
        pred_masks = np.array(pred_masks)
            
        # if self.dataset == 'OCID':
        #     refined_masks = []
        #     for pred_mask in pred_masks:
        #         refined_mask = np.where(zero_depth, False, pred_mask)
        #         refined_masks.append(refined_mask)
        #     pred_masks = np.array(refined_masks)
        time_elapsed = time.time() - start_time
        
        return pred_masks, None, time_elapsed, None
    
    