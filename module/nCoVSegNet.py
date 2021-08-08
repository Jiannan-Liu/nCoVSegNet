# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from module.ECAresnet import ECAResnet
import torchvision.models as models
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCA_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCA_1, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class GCA_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCA_2, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DAF(nn.Module):
    def __init__(self, channels_high, channels_low, kernel_size=3, upsample=True):
        super(DAF, self).__init__()
        # Global Attention Upsample
        self.conv1_1 = conv1x1(channels_high, channels_low)
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        self.conv2d = nn.Conv2d(channels_high, channels_low, kernel_size=7, stride=1, padding=3)

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

        if upsample:
            self.conv_upsample_1 = nn.ConvTranspose2d(channels_high, channels_high, kernel_size=4,  stride=2, padding=1, bias=False)
            self.conv_upsample_2 = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1,
                                                      bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fms_high, fms_low):

        b, c, h, w = fms_high.shape
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.sigmoid(fms_high_gp)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        fms_att = fms_low_mask * fms_high_gp

        fms_high_1 = self.conv_upsample_1(fms_high)
        max_out, _ = torch.max(fms_high_1, dim=1, keepdim=True)
        fms_high_gm = max_out
        fms_high_gm = self.sigmoid(fms_high_gm)
        fms_att = fms_att * fms_high_gm

        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample_2(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class nCoVSegNet(nn.Module):
    def __init__(self, n_class=1):
        super(nCoVSegNet, self).__init__()
        # ---- ResNet Backbone ----
        self.ecaresnet = ECAResnet()
        # ---- Receptive Field Block like module ----

        self.rfb1 = GCA_1(256, 128)
        self.rfb2 = GCA_1(512, 256)
        self.rfb3 = GCA_1(1024, 512)
        self.rfb4 = GCA_2(2048, 1024)

        bottom_ch = 1024
        self.gau3 = DAF(bottom_ch, 512)
        self.gau2 = DAF(bottom_ch // 2, 256)
        self.gau1 = DAF(bottom_ch // 4, 128)

        self.conv1_1 = conv1x1(128, 1)
        self.conv1_2 = conv1x1(256, 1)
        self.conv1_3 = conv1x1(512, 1)
        self.conv1_4 = conv1x1(1024, 1)

        if self.training:
            self.initialize_weights()
            print('initialize_weights')

    def forward(self, x):
        x = self.ecaresnet.conv1(x)
        x = self.ecaresnet.bn1(x)

        x = self.ecaresnet.relu(x)

        # ---- low-level features ----
        x = self.ecaresnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.ecaresnet.layer1(x)      # bs, 256, 88, 88

        # ---- high-level features ----
        x2 = self.ecaresnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.ecaresnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.ecaresnet.layer4(x3)     # bs, 2048, 11, 11

        x1_rfb = self.rfb1(x1)        # 256 -> 256
        x2_rfb = self.rfb2(x2)        # 512 -> 512
        x3_rfb = self.rfb3(x3)        # 1024 -> 1024
        x4_rfb = self.rfb4(x4)

        x3 = self.gau3(x4_rfb, x3_rfb)  # 1/16
        x2 = self.gau2(x3, x2_rfb)  # 1/8
        x1 = self.gau1(x2, x1_rfb)  # 1/4

        map_1 = self.conv1_1(x1)
        map_2 = self.conv1_2(x2)
        map_3 = self.conv1_3(x3)
        map_4 = self.conv1_4(x4_rfb)

        lateral_map_4 = F.interpolate(map_4, scale_factor=32, mode='bilinear')
        lateral_map_3 = F.interpolate(map_3, scale_factor=16, mode='bilinear')
        lateral_map_2 = F.interpolate(map_2, scale_factor=8, mode='bilinear')
        lateral_map_1 = F.interpolate(map_1, scale_factor=4, mode='bilinear')

        return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        pretrained_dict = res50.state_dict()
        model_dict = self.ecaresnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.ecaresnet.load_state_dict(model_dict)


