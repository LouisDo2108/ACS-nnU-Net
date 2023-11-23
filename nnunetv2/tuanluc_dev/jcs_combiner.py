import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.tuanluc_dev.utils import *


class FrozenBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(FrozenBatchNorm, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale

        if x.dim() == 4:
            scale = scale.view(1, -1, 1, 1)
            bias = bias.view(1, -1, 1, 1)
        elif x.dim() == 5:
            scale = scale.view(1, -1, 1, 1, 1)
            bias = bias.view(1, -1, 1, 1, 1)
        else:
            raise ValueError("Input tensor must be 4-dimensional or 5-dimensional.")

        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm(nOut)
            else:
                self.bn = nn.BatchNorm3d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels//reduction, bias=True)
        self.linear2 = nn.Linear(in_channels//reduction, in_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if x.dim() == 4:
            N, C, H, W = x.shape
            spatial_dims = (2, 3)  # dimensions representing height and width
        elif x.dim() == 5:
            N, C, D, H, W = x.shape
            spatial_dims = (2, 3, 4)  # dimensions representing depth, height, and width
        else:
            raise ValueError("Input tensor must be 4-dimensional or 5-dimensional.")

        embedding = F.adaptive_avg_pool3d(x, output_size=1) if x.dim() == 5 else F.adaptive_avg_pool2d(x, output_size=1)
        embedding = embedding.view(N, C)
        fc1 = self.act(self.linear1(embedding))
        fc2 = torch.sigmoid(self.linear2(fc1))

        if x.dim() == 4:
            fc2 = fc2.view(N, C, 1, 1)  # reshape fc2 for height and width
        else:
            fc2 = fc2.view(N, C, 1, 1, 1)  # reshape fc2 for depth, height, and width

        return x * fc2


class JCSCombiner(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = ConvBNReLU(in_channels*2, in_channels, ksize=1, pad=0, use_bn=False)
        self.se_block = SEBlock(in_channels, in_channels)
        self.conv3x3 = ConvBNReLU(in_channels, in_channels)
    
    def forward(self, cls_features, seg_features):
        x = self.conv1x1(torch.cat([cls_features, seg_features], dim=1))
        x = self.se_block(x)
        x = self.conv3x3(x)
        return x


if __name__ == "__main__":
    pass
    # # Load the classifier
    # from nnunetv2.tuanluc_dev.get_network_from_plans_dev import get_encoder

    # # Process input using the classifier
    # temp = torch.randn(4, 4, 128, 128, 128).to(torch.device("cuda"))
    # # classifier_out = classifier_encoder(temp)

    # # Load the segmenter
    # import os
    # nnunet_trainer = get_encoder()
    # model = nnunet_trainer.network
    # model(temp)
    