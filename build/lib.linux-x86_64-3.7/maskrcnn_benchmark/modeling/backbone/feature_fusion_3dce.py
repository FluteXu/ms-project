# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The original 3DCE, fusing features of neighboring slices
and only keep the feature map of the central slice"""
import torch
from collections import namedtuple
from torch import nn
from maskrcnn_benchmark.config import cfg


class FeatureFusion3dce(nn.Module):
    def __init__(self):
        super(FeatureFusion3dce, self).__init__()
        self.num_image = cfg.INPUT.NUM_IMAGES_3DCE
        self.out_dim = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.in_dim = cfg.MODEL.BACKBONE.IN_CHANNELS

        self.conv = nn.Conv2d(self.num_image * self.in_dim, self.out_dim, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, fs, images=None):
        # x = x[0].reshape(-1, self.num_image * x[0].shape[1], x[0].shape[2], x[0].shape[3])
        # x = self.conv(x)

        fused_fs = []
        for x in fs:
            x = x.reshape(-1, self.num_image * x.shape[1], x.shape[2], x.shape[3])
            x = x.to(torch.float)
            x = self.conv(x)
            fused_fs.append(x)

        if images is not None:
            # images = images[int(self.num_image/2)::self.num_image]  # only keep central ones
            images.tensors = images.tensors[int(self.num_image/2)::self.num_image]
            return fused_fs, images
        else:
            print('Error: images is None')
        #     return [x], images
        # return [x]
