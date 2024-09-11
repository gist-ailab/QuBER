import os

import numpy as np
import torch
from torchvision import transforms
from segmentation_refinement.download import download_and_or_check_model_file


def process_high_res_im(model, im, seg, L=900):

    stride = L//2

    _, _, h, w = seg.shape

    """
    Global Step
    """
    if max(h, w) > L:
        im_small = resize_max_side(im, L, 'area')
        seg_small = resize_max_side(seg, L, 'area')
    elif max(h, w) < L:
        im_small = resize_max_side(im, L, 'bicubic')
        seg_small = resize_max_side(seg, L, 'bilinear')
    else:
        im_small = im
        seg_small = seg

    images = safe_forward(model, im_small, seg_small)

    pred_224 = images['pred_224']
    pred_56 = images['pred_56_2']
    
    """
    Local step
    """

    for new_size in [max(h, w)]:
        im_small = resize_max_side(im, new_size, 'area')
        seg_small = resize_max_side(seg, new_size, 'area')
        _, _, h, w = seg_small.shape

        combined_224 = torch.zeros_like(seg_small)
        combined_weight = torch.zeros_like(seg_small)

        r_pred_224 = (F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=False)>0.5).float()*2-1
        r_pred_56 = F.interpolate(pred_56, size=(h, w), mode='bilinear', align_corners=False)*2-1

        padding = 16
        step_size = stride - padding*2
        step_len  = L

        used_start_idx = {}
        for x_idx in range((w)//step_size+1):
            for y_idx in range((h)//step_size+1):

                start_x = x_idx * step_size
                start_y = y_idx * step_size
                end_x = start_x + step_len
                end_y = start_y + step_len

                # Shift when required
                if end_y > h:
                    end_y = h
                    start_y = h - step_len
                if end_x > w:
                    end_x = w
                    start_x = w - step_len

                # Bound x/y range
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)

                # The same crop might appear twice due to bounding/shifting
                start_idx = start_y*w + start_x
                if start_idx in used_start_idx:
                    continue
                else:
                    used_start_idx[start_idx] = True
                
                # Take crop
                im_part = im_small[:,:,start_y:end_y, start_x:end_x]
                seg_224_part = r_pred_224[:,:,start_y:end_y, start_x:end_x]
                seg_56_part = r_pred_56[:,:,start_y:end_y, start_x:end_x]

                # Skip when it is not an interesting crop anyway
                seg_part_norm = (seg_224_part>0).float()
                high_thres = 0.9
                low_thres = 0.1
                if (seg_part_norm.mean() > high_thres) or (seg_part_norm.mean() < low_thres):
                    continue
                grid_images = safe_forward(model, im_part, seg_224_part, seg_56_part)
                grid_pred_224 = grid_images['pred_224']

                # Padding
                pred_sx = pred_sy = 0
                pred_ex = step_len
                pred_ey = step_len

                if start_x != 0:
                    start_x += padding
                    pred_sx += padding
                if start_y != 0:
                    start_y += padding
                    pred_sy += padding
                if end_x != w:
                    end_x -= padding
                    pred_ex -= padding
                if end_y != h:
                    end_y -= padding
                    pred_ey -= padding

                combined_224[:,:,start_y:end_y, start_x:end_x] += grid_pred_224[:,:,pred_sy:pred_ey,pred_sx:pred_ex]

                del grid_pred_224

                # Used for averaging
                combined_weight[:,:,start_y:end_y, start_x:end_x] += 1

        # Final full resolution output
        seg_norm = (r_pred_224/2+0.5)
        pred_224 = combined_224 / combined_weight
        pred_224 = torch.where(combined_weight==0, seg_norm, pred_224)

    _, _, h, w = seg.shape
    images = {}
    images['pred_224'] = F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=True)

    return images['pred_224']


class CascadePSPRGBD(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        # print('model, feats : ',self.feats)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.catconv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)  #Edited : Add a Conv2d layer to reduce dimention 4 to 3

        self.up_1 = PSPUpsample(1024, 1024+256, 512)
        self.up_2 = PSPUpsample(512, 512+64, 256)
        self.up_3 = PSPUpsample(256, 256+3, 32)
        
        self.final_28 = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.final_56 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.final_11 = nn.Conv2d(32+3, 32, kernel_size=1)
        self.final_21 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth, seg, inter_s8=None, inter_s4=None):

        images = {}

        """
        First iteration, s8 output
        """
        if inter_s8 is None:
            cat = torch.cat((x, depth), 1)  #Edited : Concat RGB and depth  -> 4channel
            p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
            p = torch.cat((p, seg, seg, seg), 1)
            # print('p  shape', p.shape)

            f, f_1, f_2 = self.feats(p) 
            p = self.psp(f)

            inter_s8 = self.final_28(p)
            r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode='bilinear', align_corners=False)
            r_inter_tanh_s8 = torch.tanh(r_inter_s8)

            images['pred_28'] = torch.sigmoid(r_inter_s8)
            images['out_28'] = r_inter_s8
        else:
            r_inter_tanh_s8 = inter_s8

        """
        Second iteration, s4 output
        """
        if inter_s4 is None:
            cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
            p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
            p = torch.cat((p, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)

            f, f_1, f_2 = self.feats(p) 
            p = self.psp(f)
            inter_s8_2 = self.final_28(p)
            r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode='bilinear', align_corners=False)
            r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)

            p = self.up_1(p, f_2)

            inter_s4 = self.final_56(p)
            r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode='bilinear', align_corners=False)
            r_inter_tanh_s4 = torch.tanh(r_inter_s4)

            images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
            images['out_28_2'] = r_inter_s8_2
            images['pred_56'] = torch.sigmoid(r_inter_s4)
            images['out_56'] = r_inter_s4
        else:
            r_inter_tanh_s8_2 = inter_s8
            r_inter_tanh_s4 = inter_s4

        """
        Third iteration, s1 output
        """
        cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
        p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
        p = torch.cat((p, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)

        f, f_1, f_2 = self.feats(p) 
        p = self.psp(f)
        inter_s8_3 = self.final_28(p)
        r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode='bilinear', align_corners=False)

        p = self.up_1(p, f_2)
        inter_s4_2 = self.final_56(p)
        r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode='bilinear', align_corners=False)
        p = self.up_2(p, f_1)
        p = self.up_3(p, x)


        """
        Final output
        """
        p = F.relu(self.final_11(torch.cat([p, x], 1)), inplace=True)
        p = self.final_21(p)

        pred_224 = torch.sigmoid(p)

        images['pred_224'] = pred_224
        images['out_224'] = p
        images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)
        images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)
        images['out_28_3'] = r_inter_s8_3
        images['out_56_2'] = r_inter_s4_2

        return images



import os

import numpy as np
import torch
from torchvision import transforms

from segmentation_refinement.models.psp.pspnet import RefinementModule
from segmentation_refinement.eval_helper import process_high_res_im, process_im_single_pass
from segmentation_refinement.download import download_and_or_check_model_file


class Refiner:
    def __init__(self, device='cpu', model_folder=None, download_and_check_model=True):
        """
        Initialize the segmentation refinement model.
        device can be 'cpu' or 'cuda'
        model_folder specifies the folder in which the model will be downloaded and stored. Defaulted in ~/.segmentation-refinement.
        """
        self.model = RefinementModule()
        self.device = device
        if model_folder is None:
            model_folder = os.path.expanduser("~/.segmentation-refinement")

        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, 'model')
        if download_and_check_model:
            download_and_or_check_model_file(model_path)

        model_dict = torch.load(model_path, map_location={'cuda:0': device})
        new_dict = {}
        for k, v in model_dict.items():
            name = k[7:] # Remove module. from dataparallel
            new_dict[name] = v
        self.model.load_state_dict(new_dict)
        self.model.eval().to(device)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

    def refine(self, image, depth, mask, fast=False, L=900):
        with torch.no_grad():
            """
            Refines an input segmentation mask of the image.

            image should be of size [H, W, 3]. Range 0~255.
            Mask should be of size [H, W] or [H, W, 1]. Range 0~255. We will make the mask binary by thresholding at 127.
            Fast mode - Use the global step only. Default: False. The speedup is more significant for high resolution images.
            L - Hyperparameter. Setting a lower value reduces memory usage. In fast mode, a lower L will make it runs faster as well.
            """
            image = self.im_transform(image).unsqueeze(0).to(self.device)
            mask = self.seg_transform((mask>127).astype(np.uint8)*255).unsqueeze(0).to(self.device)
            if len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

            if fast:
                output = process_im_single_pass(self.model, image, mask, L)
            else:
                output = process_high_res_im(self.model, image, mask, L)

            return (output[0,0].cpu().numpy()*255).astype('uint8')

