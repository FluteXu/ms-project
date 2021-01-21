# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms_7_1 as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        z_flip_prob = cfg.INPUT.Z_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        z_flip_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    """
    if cfg.INPUT.USE_AFFINE:
        if translate < 0:
            affine = T.RandomCrop(
                translate=translate,
                scale=scale,
                size_divisible=cfg.MODEL.SIZE_DIVISIBLE,
                output_size=affine_out_size
            )
        else:
            affine = T.AffineTransform(
                translate=translate,
                scale=scale,
                rotate=rotation,
                size_divisible=cfg.MODEL.SIZE_DIVISIBLE,
                output_size=affine_out_size
            )
    else:
        affine = T.Resize(min_size, max_size)    
    """
    affine = T.Resize(min_size, max_size)

    transform = T.Compose(
        [
            color_jitter,
            affine,
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.RandomZFlip(z_flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
