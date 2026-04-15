# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman

# Most of the contents of this file were taken from the EndoFM-LV GitHub repository:
# Repository: https://github.com/med-air/EndoFM-LV/tree/main
# Original file: https://github.com/med-air/EndoFM-LV/blob/main/models/timesformer.py

# We modified/simplified the function `get_vit_base_patch16_224` to better fit our use case.
# We added the functions `load_backbone_model` and `load_classification_model`.


import logging
from functools import partial
import torch
import torch.nn as nn

from model.timesformer import VisionTransformer
from model.helpers import load_pretrained, load_state_dict


def _conv_filter(state_dict, patch_size:int=16) -> dict:
    """Convert patch embedding weight from manual patchify + linear proj to conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def get_vit_base_patch16_224(cfg, **kwargs) -> nn.Module: 
    """Create the EndoFM-LV architecture with a patch size of 16 and a frame shape of 224x224."""
    patch_size = 16         # size of the patches the input image is split into, here 16x16 pixels
    train_crop_size = 224   # size of the video frames, here 224x224 pixels
    vit = VisionTransformer(img_size=train_crop_size, num_classes=cfg.MODEL.NUM_CLASSES,
                            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                            attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES,
                            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)
    vit.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
    vit.num_patches = (train_crop_size // patch_size) * (train_crop_size // patch_size)
    return vit


def load_backbone_model(cfg) -> nn.Module:
    """Load only the backbone weights into a requested ViT architecture."""
    # Initialize the requested architecture
    if cfg.DATA.TRAIN_CROP_SIZE == 224:
        vit = get_vit_base_patch16_224(cfg)
    else:
        raise ValueError(f"Unsupported train crop size: {cfg.DATA.TRAIN_CROP_SIZE}")
    # Load the backbone weights into the model architecture
    # We removed the parameters 'num_classes' (we always remove the classification head), 'in_chans' (we always input rgb-structured images), 'img_size' (not used in function)
    vit = load_pretrained(model=vit, pretrained_model=cfg.TIMESFORMER.BACKBONE_MODEL, num_frames=cfg.DATA.NUM_FRAMES, num_patches=vit.num_patches, attention_type=vit.attention_type, filter_fn=_conv_filter)
    logging.info(f"Loaded the backbone model from {cfg.TIMESFORMER.BACKBONE_MODEL}.")
    # Remove the classification head to add a custom one later
    vit.head = None
    logging.info("Removed the classification head. Please add a custom head for your specific task.")
    return vit


def load_classification_model(cfg, model:str) -> nn.Module:
    """Load backbone and classification head weights into a requested ViT architecture.
    
    Args: 
        cfg: A configuration object with dataset-specific attributes that determines the ViT architecture.
        model: The binary classification model, whose weights to load into the ViT architecture.
    
    Returns: 
        vit: The ViT architecture with the loaded weights.
    """
    # Initialize the requested architecture
    if cfg.DATA.TRAIN_CROP_SIZE == 224:
        vit = get_vit_base_patch16_224(cfg)
    else:
        raise ValueError(f"{cfg.DATA.TRAIN_CROP_SIZE} is an unsupported train crop size. We only support 224.")
    vit.head = nn.Linear(768, 2)

    # Load the state dict from the model file
    if model == "pretrained":
        checkpoint_path = cfg.TIMESFORMER.PRETRAINED_MODEL
    elif model == "finetuned":
        checkpoint_path = cfg.TIMESFORMER.FINETUNED_MODEL
    else:
        raise ValueError(f"{model} is an unsupported model type. We only support 'pretrained' and 'finetuned'.")
    backbone_param = load_state_dict(checkpoint_path, key='backbone_state_dict')
    cls_head = load_state_dict(checkpoint_path, key='state_dict')
    state_dict = {**backbone_param, **cls_head}

    # Resize time embeddings if they don't match (similar to resizing in load_pretrained)
    if ('time_embed' in state_dict) and (cfg.DATA.NUM_FRAMES != state_dict['time_embed'].size(1)):
        old_time_embed = state_dict['time_embed'].size(1)
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = torch.nn.functional.interpolate(time_embed, size=(cfg.DATA.NUM_FRAMES), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)
        new_time_embed = state_dict['time_embed'].size(1)
        logging.info(f"Resized time_embed from {old_time_embed} to {new_time_embed} frames.")

    # Since we keep the same attention type (divided_space_time) we don't need to check for it

    # The fine-tuned EndoFM-LV model's classification head has the key 'linear.'
    # We rename it to 'head.' for consistency
    for old_key in list(state_dict.keys()): 
        if old_key.startswith("linear."): 
            new_key = old_key.replace("linear.", "head.") 
            state_dict[new_key] = state_dict.pop(old_key)

    # Load the weights into the model architecture
    msg = vit.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded the {model} model from {checkpoint_path}.")
    logging.info(f"Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
    vit._load_msg = msg  # stash for downstream checks
    return vit