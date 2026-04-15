"""A collection of functions to apply the explainability method KernelSHAP."""

import torch
import numpy as np
import captum.attr._core.lime as lime_core
from captum.attr import KernelShap
from skimage.segmentation import slic


def patch_captum_for_mps():
    """Patch Captum's Lime/KernelShap to avoid float64 casting."""
    original_attribute_kwargs = lime_core.Lime._attribute_kwargs

    def patched_attribute_kwargs(self, *args, **kwargs):
        # Call original method but temporatily monkey-patch torch.Tensor.double
        original_double = torch.Tensor.double
        
        def float_instead_of_double(tensor):
            return tensor.float()
        
        # Replace `.double()` with `.float()` so it runs on MPS
        torch.Tensor.double = float_instead_of_double
        try:
            result = original_attribute_kwargs(self, *args, **kwargs)
        finally:
            # Restore original behavior to avoid global side effects
            torch.Tensor.double = original_double
        return result

    lime_core.Lime._attribute_kwargs = patched_attribute_kwargs


def apply_slic_per_frame(input_sample:torch.Tensor, n_segments:int, compactness:int):
    """
    Define input features for KernelSHAP using Simple Linear Iterative Clustering.
    SLIC is applied to each video frame to form superpixels.
    """
    # Prepare input sample
    B, C, T, H, W = input_sample.shape
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) 
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    device = input_sample.device
    imagenet_mean = imagenet_mean.to(device, dtype=torch.float32)
    imagenet_std = imagenet_std.to(device, dtype=torch.float32)
    clip = input_sample * imagenet_std + imagenet_mean 
    if clip.dim() == 5: 
        clip = clip[0]
    clip = clip.detach().cpu().numpy()
    # Apply SLIC
    segments = []
    for frame_idx in range(T):
        frame = clip[:, frame_idx, :, :]
        frame = np.transpose(frame, (1, 2, 0))   # (C, H, W) -> (H, W, C)
        frame_segments = slic(
            frame,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
        segments.append(frame_segments)
    # Get unique segment ids across the whole video
    global_segments = [] 
    offset = 0 
    for frame_idx in range(T): 
        seg = segments[frame_idx] 
        max_seg_id = seg.max()
        seg = seg + offset 
        global_segments.append(seg) 
        offset += max_seg_id + 1
    return global_segments


def apply_slic_per_video(input_sample:torch.Tensor, n_segments:int, compactness:int):
    """
    Define input features for KernelSHAP using Simple Linear Iterative Clustering.
    SLIC is applied to the video as a whole to form supervoxels.
    """
    # Prepare input sample
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) 
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    device = input_sample.device
    imagenet_mean = imagenet_mean.to(device, dtype=torch.float32)
    imagenet_std = imagenet_std.to(device, dtype=torch.float32)
    clip = input_sample * imagenet_std + imagenet_mean 
    if clip.dim() == 5: 
        clip = clip[0]
    clip = clip.detach().cpu().numpy() 
    clip = np.transpose(clip, (2, 3, 1, 0))   # (C, T, H, W) -> (H, W, T, C) so SLIC treats time as a spatial axis
    # Apply SLIC
    segments = slic(
        clip, 
        n_segments=n_segments,
        compactness=compactness,
        slic_zero=False,
        start_label=0,
        channel_axis=-1,
    )
    segments = np.transpose(segments, (2, 0, 1))   # (H, W, T) -> (T, H, W)
    return segments


def get_grid_segments(input_sample:torch.Tensor, n_segments:list[int]):
    """Define input features for KernelSHAP using a grid."""
    B, C, T, H, W = input_sample.shape
    segments = np.zeros((T, H, W), dtype=np.int32) 
    if H % n_segments[0] != 0:
        raise ValueError(f"Height has a remainder. Please change 'n_segments'.")
    if W % n_segments[1] != 0:
        raise ValueError(f"Width has a remainder. Please change 'n_segments'.")
    cell_h = H // n_segments[0]
    cell_w = W // n_segments[1]
    seg_id = 0 
    for frame in range(T):
        for i in range(n_segments[0]): 
            for j in range(n_segments[1]): 
                h_start = i * cell_h 
                h_end = (i + 1) * cell_h 
                w_start = j * cell_w 
                w_end = (j + 1) * cell_w 
                segments[frame, h_start:h_end, w_start:w_end] = seg_id 
                seg_id += 1 
    return segments


# We need to define a baseline that is used when an input feature is "not present".
# We use black as a baseline.
def get_baseline(input_sample:torch.Tensor):
    """Get the baseline in shape (C, H, W)."""
    _, C, _, H, W = input_sample.shape
    baseline = torch.zeros((C, H, W))
    # Normalize to ImageNet stats, since EndoFM-LV requires normalized inputs
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1) 
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    baseline = (baseline - imagenet_mean) / imagenet_std
    return baseline


# KernelSHAP requires a forward function 
def forward_function(
    mask, 
    video:torch.Tensor, 
    baseline:torch.Tensor, 
    model:torch.nn.Module, 
    segments:list, 
    class_idx:int, 
    device:torch.device,
):
    """ Forward function for KernelSHAP.
        
        Args:
            mask: A binary vector of shape (batch_size, num_features) indicating which features are present (1) or absent (0).
            video: The input video tensor of shape (1, C, T, H, W).
            baseline: The baseline to use when an input feature is absent of shape (C, H, W).
            model: The classification model to use for prediction.
            segments: The segmentation of pixels into input features in shape (T, H, W).
            class_idx: The index of the target class for which we want to compute the SHAP values.
            device: The device used for running KernelSHAP ('cpu', 'mps' or 'cuda').
        
        Returns:
            The model output logits of shape (1, num_classes) for the perturbed input video.
    """
    # Define input features by the resolution of raw attributions from GradCAM
    B, C, T, H, W = video.shape

    probabilities = []
    for m in mask:
        # Replace masked input features with baseline
        masked_video = video.clone()
        for frame in range(T): 
            seg = segments[frame] 
            for seg_id in np.unique(seg): 
                if m[seg_id].item() == 0: 
                    masked_video[0, :, frame, seg == seg_id] = baseline[:, seg == seg_id]
        masked_video = masked_video.to(device, dtype=torch.float32)
        # Conduct forward pass
        features = model(masked_video)[0]
        logits = model.head(features)
        probs = torch.softmax(logits, dim=1)
        # Return probability of a certain class
        probabilities.append(probs[0, class_idx])

    return torch.stack(probabilities)


def compute_kernelshap_attribution(
    model:torch.nn.Module, 
    input_sample:torch.Tensor, 
    target_class:int, 
    n_input_features:int,
    segments:list, 
    device:torch.device,
):
    """Compute KernelSHAP attribution for a given input sample and target class."""
    if device.type == 'mps':
        patch_captum_for_mps()

    input_sample = input_sample.detach().to(device, dtype=torch.float32)
    B, C, T, H, W = input_sample.shape 
    
    baseline = get_baseline(input_sample).to(device, dtype=torch.float32)
    explainer = KernelShap(forward_function)
    input_mask = torch.ones((B, n_input_features), dtype=torch.float32).to(device, dtype=torch.float32)      # all features present; (batch_size, num_features)
    baseline_mask = torch.zeros((B, n_input_features), dtype=torch.float32).to(device, dtype=torch.float32)  # all features absent; (batch_size, num_features)
    model = model.to(device, dtype=torch.float32)
    raw_attr = explainer.attribute(
        inputs=input_mask,
        baselines=baseline_mask,
        additional_forward_args=(input_sample, baseline, model, segments, target_class, device),
        n_samples=15000,   # adjust this based on good trade-off between runtime and stability of results
        show_progress=True,
    )
    return raw_attr