# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman

# Most contents of this file were taken from the EndoFM-LV GitHub repository and simplified:
# Repository: https://github.com/med-air/EndoFM-LV/tree/main
# Original file: https://github.com/med-air/EndoFM-LV/blob/main/models/helpers.py 

# We added the function 'pick_device'.

import os
import logging

import torch
import torch.nn.functional as F


def load_state_dict(checkpoint_path:str, key:str) -> dict:
    """ Load all relevant parameters of a model's state_dict from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file (e.g., a .pth file).
        key (str): The key to use for loading the state dict. 
            'student' for backbone model (original EndoFM-LV / backbone), 
            'backbone_state_dict' for backbone part of PolypDiag-finetuned model (pretrained),
            'state_dict' for classification head part of PolypDiag-finetuned model (pretrained),

    Returns:
        state_dict (dict): A Python dictionary containing the relevant model parameters.
    """
    # Load the checkpoint file
    if checkpoint_path and os.path.isfile(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        logging.info(f"No local checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError()
    
    # Get the student model weights 
    # Backbone model (original EndoFM-LV) uses key "student"
    # PolypDiag-finetuned EndoFM-LV (our pretrained model) uses "state_dict" and "backbone_state_dict" as keys
    # Our own finetuned models have similar configuration as the PolypDiag-finetuned EndoFM-LV
    state_dict = None
    if key in ['student', 'backbone_state_dict', 'state_dict']:
        if key in checkpoint: 
            state_dict = checkpoint[key] 
            logging.info(f"Found '{key}' key in checkpoint.")
        else:
            raise KeyError(f"Key '{key}' not found in checkpoint.")
        
    # If no known key found, assume checkpoint itself is the state dict
    else:
        state_dict = checkpoint
        logging.info(f"Unknown key '{key}'. Assuming checkpoint is the state dict itself.")
    
    # Remove "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    return state_dict


def load_pretrained(
    model:torch.nn.Module, 
    pretrained_model:torch.nn.Module, 
    num_frames:int=8, 
    num_patches:int=196, 
    attention_type:str='divided_space_time', 
    filter_fn=None,
) -> torch.nn.Module:
    """Load the original EndoFM-LV backbone (the pretrained foundational model)."""
    # The pretrained model has a method .state_dict() that returns a dictionary mapping layer names → tensors of parameters (all learned weights and biases)
    try:
        state_dict = load_state_dict(pretrained_model, key='student')['model']
    except:
        state_dict = load_state_dict(pretrained_model, key='student')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict) 

    # The original function accounted for different input channels, e.g. grayscale images
    # However, we only use RGB images (3 channels) and removed the additional functionality

    # We always discard the classification head
    classifier_name = 'head'
    keys_to_remove = [k for k in state_dict.keys() if k.startswith(classifier_name + ".")]
    if keys_to_remove:
        for k in keys_to_remove:
            logging.info(f"Removing {k} from pretrained model.")
            del state_dict[k]
    else:
        logging.info("Classifier bias not found in pretrained model.")

    # Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict['backbone.pos_embed'].size(1):
        pos_embed = state_dict['backbone.pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['backbone.pos_embed'] = new_pos_embed

    # Resizing time embeddings in case they don't match
    if 'backbone.time_embed' in state_dict and num_frames != state_dict['backbone.time_embed'].size(1):
        time_embed = state_dict['backbone.time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['backbone.time_embed'] = new_time_embed.transpose(1, 2)

    # Initializing temporal attention
    if attention_type == 'divided_space_time':
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn', 'temporal_attn')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1', 'temporal_norm1')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    # Loading the pretrained weights
    msg = model.load_state_dict(state_dict, strict=False)
    #print(msg)
    return model


def pick_device(choice:str, cuda_idx:int=0) -> torch.device:
    """Select the compute device."""
    if choice == "cuda":
        device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    elif choice == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif choice == "cpu":
        device = torch.device("cpu")
    else:   # 'auto'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device