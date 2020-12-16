# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from time import time
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..backbone.feature_fusion_3dce import FeatureFusion3dce


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.backbone = self.backbone.half()
        # self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.backbone.out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        if cfg.MODEL.USE_3D_FUSION:
            # the original 3DCE strategy, fuse last level feature maps
            self.feature_fuse = FeatureFusion3dce()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        # torch.cuda.synchronize()
        # s0 = time()
        features = self.backbone(images.tensors)
        # torch.cuda.synchronize()
        # print("backbone time: ", time() - s0)

        if cfg.MODEL.USE_3D_FUSION:
            features, images = self.feature_fuse(features, images)
            # import pdb; pdb.set_trace()
            # print('features[0].shape: ', features[0].shape)
            # print('rpn input images: ', images.tensors.shape)

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}

            # print('detector_losses', detector_losses)
            # print('proposal_losses', proposal_losses)
            # import pdb; pdb.set_trace()

            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
