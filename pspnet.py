import torch
from torch import nn
from torch.nn import functional as F

from models.psp import extractors
from models.sync_batchnorm import SynchronizedBatchNorm2d


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            SynchronizedBatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)
        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)
        p = p + sc
        p2 = self.conv2(p)

        return p + p2


class PSPNet(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)

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

    def forward(self, x, seg, inter_s8=None, inter_s4=None):

        images = {}

        """
        First iteration, s8 output
        """
        if inter_s8 is None:
            p = torch.cat((x, seg, seg, seg), 1)

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
            p = torch.cat((x, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)

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
        p = torch.cat((x, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)

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




class PSPNet_UOAIS(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()

        print("[DEBUG] : backbone", backend)
        self.feats = getattr(extractors, backend)(pretrained)
        # print('model, feats : ',self.feats)
        self.psp = PSPModule(psp_size, 1024, sizes)
        # self.catconv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)  #Edited : Add a Conv2d layer to reduce dimention 4 to 3

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
            # cat = torch.cat((x, depth), 1)  #Edited : C/oncat RGB and depth  -> 4channel
            # p = self.catconv(cat)   #Edited : Reduce d/imention 4 to 3
            p = torch.cat((x, depth, seg, seg, seg), 1)

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
            # cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
            # p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3

            p = torch.cat((x, depth, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)

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
        # cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
        # p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3

        p = torch.cat((x, depth, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)

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



# class PSPNet_UOAIS(nn.Module):
#     def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
#                  pretrained=True):
#         super().__init__()
#         self.feats = getattr(extractors, backend)(pretrained)
#         # print('model, feats : ',self.feats)
#         self.psp = PSPModule(psp_size, 1024, sizes)
#         self.catconv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)  #Edited : Add a Conv2d layer to reduce dimention 4 to 3

#         self.up_1 = PSPUpsample(1024, 1024+256, 512)
#         self.up_2 = PSPUpsample(512, 512+64, 256)
#         self.up_3 = PSPUpsample(256, 256+3, 32)
        
#         self.final_28 = nn.Sequential(
#             nn.Conv2d(1024, 32, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1),
#         )

#         self.final_56 = nn.Sequential(
#             nn.Conv2d(512, 32, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1),
#         )

#         self.final_11 = nn.Conv2d(32+3, 32, kernel_size=1)
#         self.final_21 = nn.Conv2d(32, 1, kernel_size=1)

#     def forward(self, x, depth, seg, inter_s8=None, inter_s4=None):

#         images = {}

#         """
#         First iteration, s8 output
#         """
#         if inter_s8 is None:
#             cat = torch.cat((x, depth), 1)  #Edited : Concat RGB and depth  -> 4channel
#             p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
#             p = torch.cat((p, seg, seg, seg), 1)
#             # print('p  shape', p.shape)

#             f, f_1, f_2 = self.feats(p) 
#             p = self.psp(f)

#             inter_s8 = self.final_28(p)
#             r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode='bilinear', align_corners=False)
#             r_inter_tanh_s8 = torch.tanh(r_inter_s8)

#             images['pred_28'] = torch.sigmoid(r_inter_s8)
#             images['out_28'] = r_inter_s8
#         else:
#             r_inter_tanh_s8 = inter_s8

#         """
#         Second iteration, s4 output
#         """
#         if inter_s4 is None:
#             cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
#             p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
#             p = torch.cat((p, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)

#             f, f_1, f_2 = self.feats(p) 
#             p = self.psp(f)
#             inter_s8_2 = self.final_28(p)
#             r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode='bilinear', align_corners=False)
#             r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)

#             p = self.up_1(p, f_2)

#             inter_s4 = self.final_56(p)
#             r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode='bilinear', align_corners=False)
#             r_inter_tanh_s4 = torch.tanh(r_inter_s4)

#             images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
#             images['out_28_2'] = r_inter_s8_2
#             images['pred_56'] = torch.sigmoid(r_inter_s4)
#             images['out_56'] = r_inter_s4
#         else:
#             r_inter_tanh_s8_2 = inter_s8
#             r_inter_tanh_s4 = inter_s4

#         """
#         Third iteration, s1 output
#         """
#         cat = torch.cat((x, depth), 1) #Edited : Concat RGB and depth  -> 4channel
#         p = self.catconv(cat)   #Edited : Reduce dimention 4 to 3
#         p = torch.cat((p, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)

#         f, f_1, f_2 = self.feats(p) 
#         p = self.psp(f)
#         inter_s8_3 = self.final_28(p)
#         r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode='bilinear', align_corners=False)

#         p = self.up_1(p, f_2)
#         inter_s4_2 = self.final_56(p)
#         r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode='bilinear', align_corners=False)
#         p = self.up_2(p, f_1)
#         p = self.up_3(p, x)


#         """
#         Final output
#         """
#         p = F.relu(self.final_11(torch.cat([p, x], 1)), inplace=True)
#         p = self.final_21(p)

#         pred_224 = torch.sigmoid(p)

#         images['pred_224'] = pred_224
#         images['out_224'] = p
#         images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)
#         images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)
#         images['out_28_3'] = r_inter_s8_3
#         images['out_56_2'] = r_inter_s4_2

#         return images