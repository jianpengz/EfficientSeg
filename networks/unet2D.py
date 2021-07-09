import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

affine_par = True
inplace = True
import functools

import sys, os

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(in_planes, out_planes, kernel_size=(3,3), stride=(1,1), padding=1, dilation=1, groups=1, bias=False, weight_std=False):
    "3x3 convolution with padding"
    if weight_std:
        return Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

# Conv-BN-Relu
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,inplace=True, weight_std=False):
        super(SeparableConv2d,self).__init__()
        self.gn0 = nn.GroupNorm(8, in_channels)
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = conv3x3(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias, weight_std=weight_std)
        self.gn1 = nn.GroupNorm(8, in_channels)
        self.pointwise = conv3x3(in_channels,out_channels,1,1,0,1,1,bias=bias, weight_std=weight_std)

    def forward(self,x):
        x = self.gn0(x)
        x = self.relu0(x)

        x = self.depthwise(x)
        x = self.gn1(x)
        x = self.pointwise(x)
        return x


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=(1,1), dilation=(1,1), downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = SeparableConv2d(inplanes,planes,3,stride=1,padding=dilation * multi_grid,dilation=dilation * multi_grid, bias=False, weight_std=weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(planes,planes,3,stride=1,padding=dilation * multi_grid,dilation=dilation * multi_grid, bias=False, weight_std=weight_std)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = x + residual

        return out



class unet(nn.Module):  ## up x 16
    def __init__(self, shape, block, layers, num_classes=3, weight_std = False):
        self.shape = shape
        self.weight_std = weight_std
        super(unet, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3(3, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2,2), weight_std=self.weight_std)
        )

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            conv3x3(32, 64, kernel_size=(3, 3), stride=(1, 1), weight_std=self.weight_std)
        )
        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3x3(64, 128, kernel_size=(3, 3), stride=(1, 1), weight_std=self.weight_std)
        )
        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3x3(128, 256, kernel_size=(3, 3), stride=(1, 1), weight_std=self.weight_std)
        )
        self.conv_256_512 = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3x3(256, 512, kernel_size=(3, 3), stride=(1, 1), weight_std=self.weight_std)
        )
        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1))
        self.layer2 = self._make_layer(block, 128,128, layers[2], stride=(1, 1))
        self.layer3 = self._make_layer(block, 256,256, layers[3], stride=(1, 1))
        self.layer4 = self._make_layer(block, 512,512, layers[4], stride=(1, 1))

        self.fusionConv = nn.Sequential(
            nn.Dropout3d(0.2),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
            conv3x3(512, 256, kernel_size=(3, 3), padding=(1,1), weight_std=self.weight_std)
        )

        self.x16_resb = self._make_layer(block, 256, 128, 1, stride=(1,1))
        self.x8_resb = self._make_layer(block, 128, 64, 1, stride=(1,1))
        self.x4_resb = self._make_layer(block, 64, 32, 1, stride=(1,1))
        self.x2_resb = self._make_layer(block, 32, 32, 1, stride=(1,1))

        self.cls_conv = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1)
        )


    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1), dilation=(1,1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3(inplanes, outplanes, kernel_size=(1, 1), stride=stride, padding=(0,0), weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def _3D_2D(self, tsr):
        bs, channels, depth, height, weight = tsr.size()
        x = tsr.transpose(1, 2).contiguous()
        x = x.view(-1, channels, height, weight)
        return x

    def _2D_3D(self, tsr, depth):
        bs_depth, channels, height, weight = tsr.size()
        x = tsr.view(bs_depth // depth, depth, channels, height, weight).transpose(1, 2)
        return x

    def forward(self, x):
        # 3D to 2D
        # _depth = x.shape[2]
        # x = self._3D_2D(x) # bs x C(3) x H x W

        # encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)
        skip4 = x

        x = self.conv_256_512(x)
        x = self.layer4(x)

        x = self.fusionConv(x)

        # decoder
        x = F.interpolate(x, size=(int(self.shape[0]/16), int(self.shape[1]/16)), mode='bilinear', align_corners=True)
        x = x + skip4
        x = self.x16_resb(x)

        x = F.interpolate(x, size=(int(self.shape[0]/8), int(self.shape[1]/8)), mode='bilinear', align_corners=True)
        x = x + skip3
        x = self.x8_resb(x)

        x = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4)), mode='bilinear', align_corners=True)
        x = x + skip2
        x = self.x4_resb(x)
        
        x = F.interpolate(x, size=(int(self.shape[0] / 2), int(self.shape[1]/ 2)), mode='bilinear', align_corners=True)
        x = x + skip1
        x = self.x2_resb(x)

        x = self.cls_conv(x)

        out = F.interpolate(x, size=(int(self.shape[0]), int(self.shape[1])), mode='bilinear', align_corners=True)

        # 2D to 3D
        # out = self._2D_3D(out, _depth)

        return out


def UNet(shape, num_classes, weight_std=False):
    model = unet(shape, NoBottleneck, [1, 1, 1, 1, 1], num_classes, weight_std)
    return model