from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from models.sync_batchnorm import SynchronizedBatchNorm2d

def load_weights_sequential(target, source_state):
    
    new_dict = OrderedDict()
    # for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
    #     print(k1, v1.shape, k2, v2.shape)
    #     new_dict[k1] = v2

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            tar_v = source_state[k1]
            if v1.shape != tar_v.shape:
                # Init the new segmentation channel with zeros
                # print(v1.shape, tar_v.shape)
                c, _, w, h = v1.shape
                tar_v = torch.cat([
                    tar_v, 
                    torch.zeros((c,3,w,h)),
                ], 1)

            new_dict[k1] = tar_v

    target.load_state_dict(new_dict)

def load_weight_wo_fist(target, source_state):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if k1 == 'conv1.weight':  # Skip the first conv layer weights
            continue  # Skip copying this layer
        if k1 not in source_state:
            print(f"Warning: '{k1}' not in source_state, using target's original weight.")
            new_dict[k1] = v1
            continue

        # Existing code logic for handling different shapes, excluding 'conv1'
        if not 'num_batches_tracked' in k1:
            tar_v = source_state[k1]
            if v1.shape != tar_v.shape:
                print(f"Shape mismatch at {k1}: target {v1.shape}, source {tar_v.shape}. Skipping.")
                continue  # Skip layers with mismatched shapes
            new_dict[k1] = tar_v

    # Load the state dict with the adjusted weights, strict=False allows skipping unmatched keys
    target.load_state_dict(new_dict, strict=False)

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)  # /2
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)  # /2

        x_2 = self.layer1(x)
        x = self.layer2(x_2)   # /2
        x = self.layer3(x)
        x = self.layer4(x)

        return x, x_1, x_2


class ResNet_UOAIS(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet_UOAIS, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)  # /2
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)  # /2

        x_2 = self.layer1(x)
        x = self.layer2(x_2)   # /2
        x = self.layer3(x)
        x = self.layer4(x)

        return x, x_1, x_2


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50_uoais(pretrained =False):
    model = ResNet_UOAIS(Bottleneck, [3, 4, 6, 3])
    print("The base model pretrained is  {}".format(pretrained))
    if pretrained:
        load_weight_wo_fist(model, model_zoo.load_url(model_urls['resnet50']))
    return model

