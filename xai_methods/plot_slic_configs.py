"""This script visualizes the SLIC-based input feature division in different configurations of SLIC hyperparameters."""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.color import rgb2lab 
from skimage.segmentation import slic, mark_boundaries

from data.process_polypgen_data import PolypGenDataset


def apply_slic_in_multiple_configs(
    input_sample:np.ndarray, 
    method:str, 
    colorspace:str, 
    compactness_levels:list[int], 
    n_segments_levels:list[int],
):
    """Define input features for KernelSHAP using Simple Linear Iterative Clustering (SLIC)."""
    T, H, W, C = input_sample.shape
    # Switch to Lab colorspace if requested
    if colorspace == "lab":
        clip = rgb2lab(input_sample)
    elif colorspace == "rgb":
        clip = input_sample
    else:
        raise ValueError("'{colorspace}' is an invalid colorspace. Use 'rgb' or 'lab'.")
    # Apply the SLIC method
    segments = []
    configs = [(n, c) for n in n_segments_levels for c in compactness_levels]
    if method == "SLIC_frame":
        # SLIC_frame applies SLIC to each video frame individually to form superpixels
        for _, config in enumerate(configs):
            config_segments = []
            for frame_idx in range(T):
                frame = clip[frame_idx]
                frame_segments = slic(
                    frame,
                    n_segments=config[0],
                    compactness=config[1],
                    start_label=0,
                    channel_axis=-1,
                )
                config_segments.append(frame_segments)
            segments.append(config_segments)
    elif method == "SLIC_clip":
        # SLIC_clip applies SLIC to the video to form supervoxels
        clip = np.transpose(clip, (1, 2, 0, 3))   # (T, H, W, C) -> (H, W, T, C) so SLIC treats time as a spatial axis
        for _, config in enumerate(configs):
            config_segments = slic(
                clip, 
                n_segments=config[0],
                compactness=config[1],
                slic_zero=False,
                start_label=0,
                channel_axis=-1,
            )
            config_segments = np.transpose(config_segments, (2, 0, 1))   # (H, W, T) -> (T, H, W)
            segments.append(config_segments)
    else:
        raise ValueError("'{method}' is an invalid SLIC method. Use 'SLIC_frame' or 'SLIC_clip'.")
    return configs, segments


def visualize_clip_in_configs(
    input_sample:np.ndarray, 
    clip_id:str, 
    label:int, 
    configs:list[tuple], 
    segments:list[np.ndarray], 
    input_feature_idx:int=None,
):
    """ Plot the input features in different SLIC hyperparameter configurations."""
    n_frames = len(input_sample)
    n_configs = len(configs)

    fig = plt.figure(figsize=(15, 20))
    fig.suptitle(f"SLIC segments for {clip_id} (class {label})", fontsize=16, y=0.93)
    gs = gridspec.GridSpec(n_configs, n_frames+1, wspace=0.02, hspace=0.5)
    for c_idx, config in enumerate(configs):
        # Add config description 
        ax_text = fig.add_subplot(gs[c_idx, 0]) 
        ax_text.axis("off") 
        ax_text.text(0.0, 0.5, f"n_segments: {config[0]}\ncompactness: {config[1]}", fontsize=12, va="center", ha="left")
        # Add frames and their segmentation
        for f_idx, frame in enumerate(input_sample):
            ax = fig.add_subplot(gs[c_idx, f_idx+1])
            slic = segments[c_idx][f_idx]
            if input_feature_idx is not None:
                # Highlight a specific superpixel to visualize its location and shape across frames
                mask = (slic == input_feature_idx) 
                frame[mask] = np.array([1.0, 1.0, 0.0])
            vis = mark_boundaries(frame, slic)
            img = ax.imshow(vis)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(f"Frame {f_idx+1}", fontsize=12)

    plt.savefig(f"clip_{clip_id}_slic_segments.png", dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize input feature division for KernelSHAP.')
    parser.add_argument('--method', type=str, choices=["SLIC_frame", "SLIC_clip"], default="SLIC_clip", help="The method used to define the input features for KernelSHAP.")
    parser.add_argument('--split', type=str, choices=["train", "val", "test"], default="val", help="The dataset split to be inspected.")
    args = parser.parse_args()

    ds = PolypGenDataset(split=args.split)
    n_input_samples = ds.__len__()
    for clip_idx in range(n_input_samples):
        # Prepare the input sample
        input_sample, label, clip_id = ds.__getitem__(idx=clip_idx)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) 
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        input_sample = input_sample * imagenet_std + imagenet_mean 
        input_sample = input_sample.permute(1, 2, 3, 0).detach().cpu().numpy()
        if input_sample.max() > 1.0:
            raise ValueError("Pixel values must be normalized to [0,1].")
        # Form input features using a set of SLIC hyperparameter configurations
        configs, segments = apply_slic_in_multiple_configs(
            input_sample, 
            method=args.method,
            colorspace="rgb",
            compactness_levels=[10,15,20],
            n_segments_levels=[10,15,20],
        )
        # Plot the input features
        visualize_clip_in_configs(input_sample, clip_id, label, configs, segments, input_feature_idx=None)