import os
import cv2
import glob
import numpy as np
import imageio
import torch
import imgviz
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict


from evaluation import multilabel_metrics
from preprocess_utils import normalize_depth, inpaint_depth
from termcolor import colored
import torch.backends.cudnn as cudnn

# append parents directory to sys.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_model import UOAISNet, UCN, UOISNet, MSMFormer, SAM, LoadNpyBaseModel, Empty, GT, Detic, GroundedSAM
from refiner_model import MaskRefiner, RICE, CascadePSP, LoadNpyRefinerModel, SAMRefiner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.random.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]
W, H = 640, 480


def run_eval(args):


    # load base model for initial mask prediction
    # !TODO: Add options for initial mask prediction
    if args.base_model == "uoaisnet":
        base_model = UOAISNet()
    elif args.base_model == "ucn":
        base_model = UCN(zoom_in=False, dataset=args.test_dataset)
    elif args.base_model == "ucn-zoomin":
        base_model = UCN(zoom_in=True, dataset=args.test_dataset)
    elif args.base_model == "uoisnet3d":
        base_model = UOISNet(dataset=args.test_dataset)
    elif args.base_model == "msmformer":
        base_model = MSMFormer(dataset=args.test_dataset, zoom_in=False)
    elif args.base_model == "msmformer-zoomin":
        base_model = MSMFormer(dataset=args.test_dataset, zoom_in=True)
    elif args.base_model == "sam":
        base_model = SAM(dataset=args.test_dataset)
    elif args.base_model == "sam-depth":
        base_model = SAM(dataset=args.test_dataset, depth_input=True)
    elif args.base_model == "npy":
        base_model = LoadNpyBaseModel(npy_folder='/SSDe/seunghyeok_back/mask-refiner/segfix/{}/initial_mask_predict_ucn_zoomin'.format(args.test_dataset))
    elif args.base_model == "empty":
        base_model = Empty()
    elif args.base_model == "gt":
        base_model = GT(dataset=args.test_dataset)
    elif args.base_model == "detic":
        base_model = Detic(dataset=args.test_dataset)
    elif args.base_model == "grounded-sam":
        base_model = GroundedSAM()
    else:
        print("Invalid base model name: {}".format(args.base_model))
        print("Available options: uoaisnet, ucn, ucn-zoomin, uoisnet3d, msmformer, msmformer-zoomin")
        raise NotImplementedError

    # load refiner model for mask refinement
    if args.refiner_model == "maskrefiner":
        refiner_model = MaskRefiner(args.config_file, 
                                weights_file = args.weights_file,
                                dataset=args.test_dataset)
    elif args.refiner_model == "cascadepsp":
        refiner_model = CascadePSP(L=900, fast=False, dataset=args.test_dataset)
    elif args.refiner_model == "cascadepsp-rgbd":
        refiner_model = CascadePSP(L=900, fast=False, dataset=args.test_dataset, depth=True)
    elif args.refiner_model == "rice":
        refiner_model = RICE(base_model=args.base_model, 
                             dataset=args.test_dataset,
                            )
    elif args.refiner_model == "sam":
        refiner_model = SAMRefiner(prompt_type='mask', dataset=args.test_dataset)
    elif args.refiner_model == "hq-sam-pretrained":
        refiner_model = SAMRefiner(prompt_type='mask', dataset=args.test_dataset, hq=True, pretrained=True)
    elif args.refiner_model == "hq-sam":
        refiner_model = SAMRefiner(prompt_type='mask', dataset=args.test_dataset, hq=True, pretrained=False)
        # print number of parameters
        print("Number of parameters: ", sum(p.numel() for p in refiner_model.sam.parameters()))
    elif args.refiner_model == "save":
        refiner_model = None
    elif args.refiner_model == "npy":
        refiner_model = LoadNpyRefinerModel(npy_folder='/SSDe/seunghyeok_back/mask-refiner/segfix/{}/label_w_segfix_detic_20_0.6'.format(args.test_dataset, args.dt_max_distance, args.mask_threshold),
                                            dataset=args.test_dataset)
    else:
        print("Invalid refiner model name: {}".format(args.refiner_model))
        print("Available options: maskrefiner, cascadepsp, rice")
        raise NotImplementedError

    # load dataset
    if args.test_dataset == "OSD":
        args.dataset_path = 'detectron2_datasets/OSD-0.2-depth'
        rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
        depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
        anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} visible masks".format(
                    len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))

    elif args.test_dataset == "WISDOM":
        args.dataset_path = 'detectron2_datasets/wisdom-real/high-res'
        test_indices = np.load(os.path.join(args.dataset_path, 'test_indices.npy'))
        rgb_paths = [os.path.join(args.dataset_path, 'color_ims', 'image_{:06d}.png'.format(i)) for i in test_indices]
        depth_paths = [os.path.join(args.dataset_path, 'depth_ims_numpy', 'image_{:06d}.npy'.format(i)) for i in test_indices]
        anno_paths = [os.path.join(args.dataset_path, 'modal_segmasks', 'image_{:06d}.png'.format(i)) for i in test_indices]
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print(colored("Evaluation on WISDOM dataset: {} rgbs, {} depths, {} visible masks".format(
                    len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))


    elif args.test_dataset == "OCID":
        args.dataset_path = 'detectron2_datasets/OCID-dataset'
        rgb_paths = []
        depth_paths = []
        anno_paths = []
        # load ARID20
        print("... load dataset [ ARID20 ]")
        data_root = args.dataset_path + "/ARID20"
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                seq_list = sorted(os.listdir(os.path.join(data_root, dir_1, dir_2)))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        rgb_path = os.path.join(data_root, dir_1, dir_2, seq, "rgb", data_name)
                        rgb_paths.append(rgb_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, seq, "label", data_name)
                        anno_paths.append(anno_path)
        # load YCB10
        print("... load dataset [ YCB10 ]")
        data_root = args.dataset_path +  "/YCB10"
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        c_c_m = ["cuboid", "curved", "mixed"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                for dir_3 in c_c_m:
                    seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                    for seq in seq_list:
                        data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                        if not os.path.isdir(data_dir): continue
                        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                        for data_name in data_list:
                            rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                            rgb_paths.append(rgb_path)
                            depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                            depth_paths.append(depth_path)
                            anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                            anno_paths.append(anno_path)
        # load ARID10
        print("... load dataset [ ARID10 ]")
        data_root =  args.dataset_path + "/ARID10"
        f_or_t = ["floor", "table"]
        b_or_t = ["bottom", "top"]
        c_c_m = ["box", "curved", "fruits", "mixed", "non-fruits"]
        for dir_1 in f_or_t:
            for dir_2 in b_or_t:
                for dir_3 in c_c_m:
                    seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                    for seq in seq_list:
                        data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                        if not os.path.isdir(data_dir): continue
                        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                        for data_name in data_list:
                            rgb_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                            rgb_paths.append(rgb_path)
                            depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                            depth_paths.append(depth_path)
                            anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                            anno_paths.append(anno_path)
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print(colored("Evaluation on OCID dataset: {} rgbs, {} depths, {} visible_masks".format(
                        len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))
    
    elif args.test_dataset == "HOPE":
        args.dataset_path = 'detectron2_datasets/hope_preprocessed'
        rgb_paths = sorted(glob.glob("{}/rgb/*.png".format(args.dataset_path)))
        depth_paths = sorted(glob.glob("{}/depth/*.png".format(args.dataset_path)))
        anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print(colored("Evaluation on HOPE dataset: {} rgbs, {} depths, {} visible masks".format(
                    len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))

    elif args.test_dataset == "DoPose":
        args.dataset_path = 'detectron2_datasets/DoPose'
        # every 50 frames
        rgb_paths = sorted(glob.glob("{}/rgb/*.png".format(args.dataset_path)))
        depth_paths = sorted(glob.glob("{}/depth/*.png".format(args.dataset_path)))
        anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))
        assert len(rgb_paths) == len(depth_paths)
        assert len(rgb_paths) == len(anno_paths)
        print(colored("Evaluation on DoPose dataset: {} rgbs, {} depths, {} visible masks".format(
                    len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))

    else:
        print(colored("Error: dataset {} is not supported".format(args.test_dataset), "red"))
        print("Supported datasets: OSD, OCID, HOPE, DoPose")
        raise NotImplementedError
        
    # idx = 1114
    # rgb_paths = rgb_paths[idx:]
    # depth_paths = depth_paths[idx:]
    # anno_paths = anno_paths[idx:]

    initial_metrics_all = []
    refined_metrics_all = []
    initial_pred_times = []
    refined_pred_times = []
    for rgb_path, depth_path, anno_path in zip(tqdm(rgb_paths), depth_paths, anno_paths):
        # if os.path.basename(rgb_path) in ['learn25.png', 'learn26.png', 'test25.png', 'test26.png', 'test49.png']:
            # continue    

        # load annotations
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        if args.test_dataset == "OCID":
            if "floor" in rgb_path:
                floor_table = "floor"
            elif "table" in rgb_path:
                floor_table = "table"
            for label in BG_LABELS[floor_table]:
                anno[anno == label] = 0         
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]

        # initial prediction

        initial_masks, fg_mask, initial_pred_time = base_model.predict(rgb_path, depth_path)
        initial_pred_times.append(initial_pred_time)

        if refiner_model is None:

            initial_masks = [np.where(mask > False, 255, 0) for mask in initial_masks]
            # save
            os.makedirs('segfix/{}/image'.format(args.test_dataset), exist_ok=True)
            os.makedirs('segfix/{}/inital_mask_predict'.format(args.test_dataset), exist_ok=True)
            os.makedirs('segfix/{}/label'.format(args.test_dataset), exist_ok=True)
            cv2.imwrite('segfix/{}/image/{}'.format(args.test_dataset, os.path.basename(rgb_path)), cv2.imread(rgb_path))
            cv2.imwrite('segfix/{}/label/{}'.format(args.test_dataset, os.path.basename(rgb_path)), cv2.imread(rgb_path))
            np.save('segfix/{}/inital_mask_predict/{}'.format(args.test_dataset, os.path.basename(rgb_path).replace('.png', '.npy')), initial_masks)
            # move to next for loop
            continue
                        


        # refined prediction
        # if len(initial_masks) == 0:
        #     refined_masks = initial_masks
        #     refined_output = None
        # else:
        refined_masks, refined_output, refined_pred_time, fg_mask = refiner_model.predict(rgb_path, depth_path, initial_masks, fg_mask)
        refined_pred_times.append(refined_pred_time)

        # convert it to pred for evaluation
        initial_pred = np.zeros_like(anno)
        for i, mask in enumerate(initial_masks):
            initial_pred[mask > False] = i+1            
        refined_pred = np.zeros_like(anno)
        for i, mask in enumerate(refined_masks):
            refined_pred[mask > False] = i+1
            
        if args.visualize:
            vis_dir = os.path.join(args.vis_dir, args.test_dataset, args.base_model + '_' + args.refiner_model)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.resize(rgb_img, (W, H))
            if '.npy' in depth_path:
                depth_img = np.load(depth_path)
            else:
                depth_img = imageio.imread(depth_path)
            depth_img = normalize_depth(depth_img)
            depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_img = inpaint_depth(depth_img)

            initial_vis = imgviz.instances2rgb(rgb_img.copy(), masks=initial_masks, labels=list(range(initial_masks.shape[0])), line_width=0, boundary_width=3)
            gt_vis = imgviz.instances2rgb(rgb_img.copy(), masks=[anno == label for label in labels_anno], labels=labels_anno, line_width=0, boundary_width=3)   
            refine_vis = imgviz.instances2rgb(rgb_img.copy(), masks=refined_masks, labels=list(range(refined_masks.shape[0])), line_width=0, boundary_width=3)
            refine_vis_depth = imgviz.instances2rgb(depth_img.copy(), masks=refined_masks, labels=list(range(refined_masks.shape[0])), line_width=0, boundary_width=3)
            vis_all = [rgb_img, depth_img, gt_vis, initial_vis, refine_vis, refine_vis_depth]
            # draw eee
            if refined_output is not None:
                if 'eee_boundary' in refined_output.keys():
                    eee_boundary_vis = rgb_img.copy()
                    eee_boundary = refined_output['eee_boundary'] # 4, H, W
                    eee_boundary = eee_boundary.argmax(0, keepdim=True) # 1, H, W
                    eee_boundary = torch.cat([eee_boundary == i for i in range(4)], dim=0) # 4, H, W
                    eee_boundary = eee_boundary.detach().cpu().numpy() # 4, H, W
                    eee_boundary_vis[eee_boundary[0] > False] = [0, 255, 0] # TP
                    eee_boundary_vis[eee_boundary[2] > False] = [0, 0, 255] # FP
                    eee_boundary_vis[eee_boundary[3] > False] = [255, 0, 0] # FN
                    vis_all.append(eee_boundary_vis)
                if 'eee_mask' in refined_output.keys():
                    eee_mask_vis = rgb_img.copy()
                    eee_mask = refined_output['eee_mask']
                    eee_mask = eee_mask.argmax(0, keepdim=True) # 1, H, W
                    eee_mask = torch.cat([eee_mask == i for i in range(4)], dim=0) # 2, H, W
                    eee_mask = eee_mask.detach().cpu().numpy() # 2, H, W
                    eee_mask_vis[eee_mask[0] > False] = [0, 255, 0] # TP
                    eee_mask_vis[eee_mask[2] > False] = [0, 0, 255] # FP
                    eee_mask_vis[eee_mask[3] > False] = [255, 0, 0] # FN
                    vis_all.append(eee_mask_vis)
            if fg_mask is not None:
                fg_vis = rgb_img.copy()
                fg_vis[fg_mask > False] = 0.7 * np.array([0, 255, 0]) + 0.3 * fg_vis[fg_mask > False]
                vis_all.append(fg_vis)
            cv2.imwrite(os.path.join(vis_dir, os.path.basename(rgb_path)), imgviz.tile(vis_all, border=(255, 255, 255)))
            # cv2.imwrite(os.path.basename(rgb_path), imgviz.tile(vis_all, border=(255, 255, 255)))

        # evaluate
        initial_metrics = multilabel_metrics(initial_pred, anno, 1, 1)
        refined_metrics = multilabel_metrics(refined_pred, anno, 1, 1)
        initial_metrics_all.append(initial_metrics)
        refined_metrics_all.append(refined_metrics)
        print(initial_metrics['obj_detected_075_percentage_normalized'], refined_metrics['obj_detected_075_percentage_normalized'])
    refined_pred_times = refined_pred_times[1:]
    avg_pred_time = np.sum(refined_pred_times) / len(refined_pred_times)
    std_pred_time = np.std(refined_pred_times)
    print("Average Prediction Time: {:.2f} ms".format(avg_pred_time * 1000))
    print("Std Prediction Time: {:.2f} ms".format(std_pred_time * 1000))
    
    # sum the values with same keys
    for i, metrics_all in enumerate([initial_metrics_all, refined_metrics_all]):
        result = {}
        num = len(metrics_all)
        for metrics in metrics_all:
            for k in metrics.keys():
                result[k] = result.get(k, 0) + metrics[k]
        for k in sorted(result.keys()):
            result[k] /= num

        print('\n')
        str = "Initial Masks ({})".format(args.base_model) if i == 0 else "Refined Masks ({})".format(args.refiner_model)
        print(colored("Visible Metrics for " + str, "green", attrs=["bold"]))

        print(colored("---------------------------------------------", "green"))
        print("    Overlap    |    Boundary")
        print("  P    R    F  |   P    R    F  |  %75")
        print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} ".format(
            result['Objects Precision']*100, result['Objects Recall']*100, 
            result['Objects F-measure']*100,
            result['Boundary Precision']*100, result['Boundary Recall']*100, 
            result['Boundary F-measure']*100,
            result['obj_detected_075_percentage']*100
        ))
        print(colored("---------------------------------------------", "green"))
        print(" Overlap (OSN)| Boundary (OSN)")
        print("  P    R    F |   P    R    F  |  %75")
        print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} ".format(
            result['Objects OSN Precision']*100, result['Objects OSN Recall']*100,
            result['Objects OSN F-measure']*100,
            result['Boundary OSN Precision']*100, result['Boundary OSN Recall']*100,
            result['Boundary OSN F-measure']*100,
            result['obj_detected_075_percentage_normalized']*100
        ))
        print(colored("---------------------------------------------", "green"))
        print("obj mIOU", result['obj_mIOU'])
        print("obj mIOU OSN", result['obj_mIOU_osn'])
        if i == 0:
            avg_pred_time = np.sum(initial_pred_times) / len(initial_pred_times)
            std_pred_time = np.std(initial_pred_times)
        else:
            avg_pred_time = np.sum(refined_pred_times) / len(refined_pred_times)
            std_pred_time = np.std(refined_pred_times)
        print("Average Prediction Time: {:.2f} ms".format(avg_pred_time * 1000))
        print("Std Prediction Time: {:.2f} ms".format(std_pred_time * 1000))
        try:
            save_csv(result=result, args=args, i=i)
        except:
            pass

def save_csv(result, args, i):
    od = OrderedDict()

    od['base_model'] = args.base_model
    od['test_dataset'] = args.test_dataset
    od['config_file'] = args.config_file
    od['i'] = i

    od['Objects Precision'] = result['Objects Precision']*100
    od['Objects Recall'] = result['Objects Recall']*100
    od['Objects F-measure'] = result['Objects F-measure']*100
    od['Boundary Precision'] = result['Boundary Precision']*100
    od['Boundary Recall'] = result['Boundary Recall']*100
    od['Boundary F-measure'] = result['Boundary F-measure']*100
    od['obj_detected_075_percentage'] = result['obj_detected_075_percentage']*100

    od['Objects OSN Precision'] = result['Objects OSN Precision']*100
    od['Objects OSN Recall'] = result['Objects OSN Recall']*100,
    od['Objects OSN F-measure'] = result['Objects OSN F-measure']*100
    od['Boundary OSN Precision'] = result['Boundary OSN Precision']*100
    od['Boundary OSN Recall'] = result['Boundary OSN Recall']*100
    od['Boundary OSN F-measure'] = result['Boundary OSN F-measure']*100
    od['obj_detected_075_percentage_normalized'] = result['obj_detected_075_percentage_normalized']*100

    od['mIOU'] = result["obj_mIOU"]
    od['mIOU OSN'] = result["obj_mIOU_osn"]

    pf = pd.DataFrame(od)
    config_path_list = args.config_file.split('/')
    config_name = config_path_list[-1].split('.')[0]
    csv_path = os.path.join('output', config_path_list[1], config_path_list[2], config_path_list[3], config_name, '{}_{}_eval_results.csv'.format(args.base_model, args.test_dataset))
    pf.to_csv(csv_path, index=False)

    return 0