# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import logging

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from inclearn.lib.logger import LOGGER

log = LOGGER
logger = log.LOGGER


@register_model
def deit_my_small_patch8_pure(pretrained=False, **kwargs):  # for miniImageNet, imageSize=96
    model = VisionTransformer(
        patch_size=8, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=84,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch4_pure(pretrained=False, **kwargs):  # for cifar100, imageSize=96
    print("Received kwargs:", kwargs)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    print("Received kwargs after cleanup:", kwargs)

    model = VisionTransformer(
        patch_size=3, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        img_size=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def deit_my_small_patch16_pure_224(pretrained=False, chkpt_path=None, **kwargs): # for cub200
    print("Received kwargs:", kwargs)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    print("Received kwargs after cleanup:", kwargs)
    model = VisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if chkpt_path is not None:
        chkpt = torch.load(chkpt_path, 'cpu')
        err = model.load_state_dict(chkpt, strict=False)
        logger.info(f'Loading chkpt at {chkpt_path}, status = {err}')
    return model
