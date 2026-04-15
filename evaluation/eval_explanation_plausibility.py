"""A collection of functions to evaluate plausibility of explanations for a single dataset-model-setting."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional, Tuple

from data.process_polypgen_data import PolypGenDataset, load_dataset_split


def get_masks(dataset:PolypGenDataset, polyp_class_only:bool=True) -> Tuple[List[str], List[np.ndarray]]:
    """Load segmentation masks for a PolypGen dataset.
    
        Args:
            dataset: A PolypGen datset to get the masks from.
            polyp_class_only: Whether to skip healthy-class samples.
        Returns:
            video_ids: The subclip IDs of the loaded masks, in shape (n_samples,).
            masks: The segmentation masks, in shape (n_samples, T, H, W).
    """
    n_samples = len(dataset)
    video_ids = []
    masks = []

    for idx in range(n_samples):
        _, label, video_id, mask = dataset[idx]

        # Skip healthy-class samples if requested
        if polyp_class_only and int(label) == 0:
            continue
            
        if mask.ndim == 4:
            mask = np.squeeze(mask, axis=-1)
        if mask.ndim != 3:
            raise ValueError(f"Expected mask with shape (T, H, W), got {mask.shape}.")

        # Save segmentation mask
        video_ids.append(video_id)
        masks.append(mask)
    
    return video_ids, masks


def get_saliency_maps(dataset:PolypGenDataset, dataset_name:str, split:str, model:str, xai_method:str, polyp_class_only:bool=True) -> Tuple[List[str], List[np.ndarray]]:
    """Load saliency maps for a PolypGen dataset.
    
        Args:
            dataset: A PolypGen datset to get the saliency maps from.
            dataset_name: The name of the PolypGen dataset (e.g. 'PolypGen6').
            split: The data split (e.g. 'test').
            model: The type of classification model (e.g. 'finetuned').
            xai_method: The type of explainability method (e.g. 'kernelshap').
            polyp_class_only: Whether to skip healthy-class samples.
        Returns:
            video_ids: The subclip IDs of the loaded saliency maps, in shape (n_samples,).
            saliency_maps: The saliency maps, in shape (n_samples, T, H, W).
    """
    n_samples = len(dataset)
    video_ids = []
    saliency_maps = []

    base_path = f"./xai_methods/saliency_maps/{model}_{dataset_name}/{split}"

    for idx in range(n_samples):
        _, label, video_id, _ = dataset[idx]

        # Skip healthy-class samples if requested
        if polyp_class_only and int(label) == 0:
            continue

        # Load saliency map
        saliency_map = np.load(f"{base_path}/{video_id}/polyp/attr_{xai_method}.npy")
        if saliency_map.ndim == 4:
            saliency_map = np.squeeze(saliency_map, axis=-1)
        if saliency_map.ndim != 3:
            raise ValueError(f"Expected saliency map with shape (T, H, W), got {saliency_map.shape}.")

        video_ids.append(video_id)
        saliency_maps.append(saliency_map)

    return video_ids, saliency_maps


def get_masks_and_saliency_maps(split:str, model:str, xai_method:str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load masks and saliency maps for a PolypGen dataset.
    
        Args:
            split: The data split (e.g. 'test').
            model: The type of classification model (e.g. 'finetuned').
            xai_method: The type of explainability method (e.g. 'kernelshap').
        Returns:
            masks: The segmentation masks, in shape (n_samples, T, H, W).
            saliency_maps: The saliency maps, in shape (n_samples, T, H, W).
            video_ids_M: The subclip IDs of the loaded masks and saliency maps, in shape (n_samples,).
    """
    if split not in ["val", "test"]:
        raise ValueError(f"'{split}' is an invalid split; use 'val' or 'test'.")
    if model not in ['pretrained', 'finetuned']:
        raise ValueError(f"'{model}' is an invalid model; use 'pretrained' or 'finetuned'.")
    if xai_method not in ['kernelshap', 'gradcam']:
        raise ValueError(f"'{xai_method}' is an invalid xai_method; use 'kernelshap' or 'gradcam'.")

    # Load the masks and saliency maps   
    ds = load_dataset_split(dataset="polypGen6", split=split, load_masks=True)
    video_ids_M, masks = get_masks(dataset=ds, polyp_class_only=True)
    video_ids_S, saliency_maps = get_saliency_maps(dataset=ds, dataset_name="polypGen6", split=split, model=model, xai_method=xai_method, polyp_class_only=True)
    
    # Sanity check whether the order of samples' mask and saliency map is identical
    if video_ids_M != video_ids_S:
        raise RuntimeError("Mask and saliency map order is inconsistent.")

    return masks, saliency_maps, video_ids_M


def normalize_saliency_map(saliency_map:np.ndarray) -> np.ndarray:
    """Normalize the saliency map to [0,1] range. Consider only positive attributions.
    
        Args:
            saliency_map: A sample's raw saliency map, in shape (T, H, W).
        Returns:
            normalized_saliency_map: Tne normalized saliency map, in shape (T, H, W).
    """
    saliency_map = saliency_map.copy().astype(np.float32)
    # Ignore all negative attribution scores
    saliency_map[saliency_map < 0.0] = 0.0
    # Normalize using the maximal attribution in the clip
    min_val = saliency_map.min() 
    max_val = saliency_map.max()
    if max_val > min_val:
        normalized_saliency_map = (saliency_map - min_val) / (max_val - min_val) 
    else: 
        normalized_saliency_map = np.zeros_like(saliency_map)
    return normalized_saliency_map


def compute_iou(saliency_map:np.ndarray, mask:np.ndarray, threshold:float=0.5) -> float:
    """Compute the Intersection over Union (IoU) between the highly salient region and the mask. \
    Checks how strongly the two masks agree on the polyp region.
    
        Args:
            saliency_map: Normalized saliency map of a single frame, in shape (H, W).
            mask: Binary mask of the clinically relevant area (1:polyp, 0:background), in shape (H, W).
            threshold: Threshold to binarize saliency map.
        Returns:
            iou: The IoU score in range [0,1].
    """
    # Get the highly salient region by applying the threshold
    highly_salient_region = (saliency_map >= threshold).astype(np.uint8)
    # Ensure mask is binary
    clinically_relevant_region = (mask > 0).astype(np.uint8)
    # Compute IoU score
    intersection = np.logical_and(highly_salient_region, clinically_relevant_region).sum()
    union = np.logical_or(highly_salient_region, clinically_relevant_region).sum()
    if union == 0:
        return 0.0
    iou = float(intersection/union)
    return iou


def compute_iou_curve(saliency_map:np.ndarray, mask:np.ndarray, thresholds:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute IoU across multiple thresholds.
    
        Args:
            saliency_map: Normalized saliency map of a single frame, in shape (H, W).
            mask: Binary mask of the clinically relevant area (1:polyp, 0:background), in shape (H, W).
            thresholds: The thresholds to test, in shape (n_thresholds,). If no thresholds provided, every increment of 0.05 will be tested.
        Returns:
            thresholds: The thresholds tested and their respective IoU scores, in shape (n_thresholds,).
            ious: The frame's IoU score for all thresholds tested, in shape (n_thresholds,).
    """
    if thresholds is None:
        # Test every increment of 0.05
        thresholds = np.linspace(0, 1, 21)
    ious = np.array([compute_iou(saliency_map, mask, t) for t in thresholds])
    return thresholds, ious


def find_best_threshold(model:str, xai_method:str, thresholds:Optional[np.ndarray]=None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Get the best threshold from the mean IoU curve (averaged across IoU scores of all PolypGen6 validation samples).
    
        Args:
            model: The type of classification model used for generating the polyp predictions (e.g. 'finetuned').
            xai_method: The type of XAI method used for generating the explanations (e.g. 'kernelshap').
            thresholds: The thresholds to test; must be in [0.0, 1.0]. If no thresholds provided, every increment of 0.05 will be tested.
        Returns:
            best_threshold: The best threshold across data samples.
            thresholds: The thresholds tested, in shape (n_thresholds,).
            mean_iou_curve: The mean IoU score (across samples) for all thresholds, in shape (n_thresholds,).
            std_iou_curve: The standard deviation (across samples) for all thresholds, in shape (n_thresholds,).
            all_sample_ious: Each sample's mean IoU score for all thresholds, in shape (n_samples, n_thresholds).
    """
    if model not in ["finetuned", "pretrained"]:
        raise ValueError(f"'{model}' is an invalid model; use 'finetuned' or 'pretrained.")
    if xai_method not in ["kernelshap", "gradcam"]:
        raise ValueError(f"'{xai_method}' is an invalid xai_method; use 'kernelshap' or 'gradcam'.")
    if (thresholds is not None) and (np.any((thresholds < 0.0) | (thresholds > 1.0))):
        raise ValueError(f"'{thresholds}' contains invalid values; all thresholds must be between 0.0 and 1.0.")

    # Load the masks and saliency maps
    masks, saliency_maps, _ = get_masks_and_saliency_maps(split="val", model=model, xai_method=xai_method)
    # Find the best threshold for plausibility evaluation
    all_sample_ious = []
    n_val_samples = len(saliency_maps)
    for idx in range(n_val_samples):
        # Normalize the saliency map
        norm_saliency_map = normalize_saliency_map(saliency_maps[idx])
        sample_ious = []
        for f_idx, frame in enumerate(norm_saliency_map):
            # Get the IoU curve for the frame
            thresholds, ious = compute_iou_curve(frame, masks[idx][f_idx], thresholds)
            sample_ious.append(ious)
        # Get the average IoU curve for the sample across frames
        all_sample_ious.append(np.mean(sample_ious, axis=0))
    # Get the average IoU curve and std across all validation samples
    mean_iou_curve = np.mean(all_sample_ious, axis=0)
    std_iou_curve = np.std(all_sample_ious, axis=0)
    # Best threshold is the one with the highest mean IoU
    best_idx = np.argmax(mean_iou_curve) 
    best_threshold = float(thresholds[best_idx])
    return best_threshold, thresholds, mean_iou_curve, std_iou_curve, all_sample_ious


def evaluate_threshold_dependency(
    sample_iou_curves:List[np.ndarray],
    mean_iou_curve:np.ndarray,
    std_iou_curve:np.ndarray,
    thresholds:np.ndarray,
    save_path:str,
) -> None:
    """Plot individual IoU curves of each sample, together with the dataset mean and standard deviation.
    
        Args:
            sample_iou_curves: Each sample's mean IoU score for all thresholds, in shape (n_samples, n_thresholds).
            mean_iou_curve: The mean IoU score (across samples) for all thresholds, in shape (n_thresholds,).
            std_iou_curve: The standard deviation (across samples) for all thresholds, in shape (n_thresholds,).
            thresholds: The thresholds tested, in shape (n_thresholds,).
            save_path: Output path for the plot.
    """
    sample_iou_curves = np.asarray(sample_iou_curves, dtype=np.float32)
    mean_iou_curve = np.asarray(mean_iou_curve, dtype=np.float32)
    std_iou_curve = np.asarray(std_iou_curve, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if sample_iou_curves.ndim != 2:
        raise ValueError(f"sample_iou_curves must be 2D with shape (N, n_thresholds), got {sample_iou_curves.shape}.")
    n_samples, n_thresholds = sample_iou_curves.shape
    for name, arr in [
        ("mean_iou_curve", mean_iou_curve),
        ("std_iou_curve", std_iou_curve),
        ("thresholds", thresholds),
    ]:
        if arr.shape != (n_thresholds,):
            raise ValueError(f"{name} must have shape ({n_thresholds},), got {arr.shape}.")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw each individual sample's IoU curve
    for idx in range(n_samples):
        ax.plot(thresholds, sample_iou_curves[idx], color="#9aa0a6", alpha=0.35, linewidth=1.0)

    # Draw mean IoU curve and its std
    lower = np.clip(mean_iou_curve - std_iou_curve, 0.0, 1.0)
    upper = np.clip(mean_iou_curve + std_iou_curve, 0.0, 1.0)
    ax.fill_between(thresholds, lower, upper, color="#1f77b4", alpha=0.2, label="std")
    ax.plot(thresholds, mean_iou_curve, color="#1f77b4", linewidth=2.5, label=r"mean IoU$_{clip}$ across val samples")

    # Add descriptions
    ax.set_xlabel("threshold")
    ax.set_ylabel(r"IoU$_{clip}$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Threshold Dependency of Explanation Plausibility")
    ax.legend(loc="best", fontsize=8 if n_samples <= 10 else None)

    # Save plot
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_plausibility_score(saliency_map:np.ndarray, mask:np.ndarray, threshold:float=0.5) -> Tuple[float, float, np.ndarray]:
    """Get the mean IoU score for a test input sample and its standard deviation across video frames.
    
        Args:
            saliency_map: Raw attributions for the input clip, in shape (T, H, W).
            mask: Binary masks for all frames of the input clip, in shape (T, H, W).
            threshold: Threshold to binarize saliency map.
        Returns:
            mean_iou: The clip-level IoU score, averaged from frame-level IoU scores.
            std_iou: The standard deviation of the IoU score across frames.
            sample_ious: The IoU scores for each frame.
    """
    # Normalize the saliency map
    norm_saliency_map = normalize_saliency_map(saliency_map)
    # Get the frame-level IoU scores
    sample_ious = []
    for idx, frame in enumerate(norm_saliency_map):
        # Compute the IoU only for frames that contain a polyp (ignore "empty" masks)
        if mask[idx].sum() != 0:
            sample_ious.append(compute_iou(frame, mask[idx], threshold))
        else:
            sample_ious.append(np.nan)
    # Get the mean IoU and its standard deviation across frames
    sample_ious = np.array(sample_ious)
    mean_iou = float(np.nanmean(sample_ious)) 
    std_iou = float(np.nanstd(sample_ious))
    return mean_iou, std_iou, sample_ious


def evaluate_frame_dependency(ious_per_frame:List[np.ndarray], save_path:str) -> None:
    """Plot the IoU score progression over frames for all PolypGen6 test samples, \
    together with the dataset mean and std.
    
        Args:
            ious_per_frame: The IoU scores for each frame of each sample, in shape (n_samples, n_frames).
            save_path: Output path for the plot.
    """
    n_samples = len(ious_per_frame)
    n_frames = len(ious_per_frame[0])
    frame_indices = np.arange(1, n_frames+1)

    # Compute the mean IoU and std per frame
    arr = np.array(ious_per_frame, dtype=float)
    iou_mean_per_frame = np.nanmean(arr, axis=0)
    iou_std_per_frame  = np.nanstd(arr, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw each individual sample curve
    for idx in range(n_samples):
        ax.plot(frame_indices, ious_per_frame[idx], color="#9aa0a6", alpha=0.35, linewidth=1.0)

    # Draw mean IoU curve and its std
    lower = np.clip(iou_mean_per_frame - iou_std_per_frame, 0.0, 1.0)
    upper = np.clip(iou_mean_per_frame + iou_std_per_frame, 0.0, 1.0)
    ax.fill_between(frame_indices, lower, upper, color="#1f77b4", alpha=0.2, label="std")
    ax.plot(frame_indices, iou_mean_per_frame, color="#1f77b4", linewidth=2.5, label=r"mean IoU$_{frame}$ across test samples")

    # Add descriptions
    ax.set_xlabel("Frame")
    ax.set_ylabel(r"IoU$_{frame}$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Consistency of Explanation Plausibility across Frames")
    ax.legend(loc="best")

    # Save plot
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight") 
    plt.close(fig)


def evaluate_polyp_prob_relation(video_ids:List[str], iou_means:List[float], meta_path:str, save_path:str) -> List[float]:
    """Plot the mean IoU score against the polyp-class prediction probability for all PolypGen6 test samples.
    
        Args:
            video_ids: The subclip IDs of each sample, in shape (n_samples,).
            iou_means: The mean IoU of each sample, in shape (n_samples,).
            meta_path: A .csv table that stores metadata, including the polyp class prediction probability of each sample.
            save_path: Output path for the plot.
        Returns:
            polyp_preds: The polyp class prediction probability of each sample, in shape (n_samples,).
    """
    # Get the polyp class prediction probabilities
    metadata = pd.read_csv(meta_path, sep=",")
    polyp_preds = []
    for video_id in video_ids:
        polyp_pred = metadata.loc[metadata["video_id"]==video_id, "prob_polyp_percent"].values[0]
        polyp_preds.append(polyp_pred)
    # Build the data frame
    df = pd.DataFrame({
        "video_id": video_ids,
        "polyp_pred": polyp_preds,
        "mean_iou": iou_means,
    })
    # Plot the data
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="polyp_pred",
        y="mean_iou",
    )
    plt.title(f"Relationship between Model Confidence and Explanation Plausibility")
    plt.xlabel("polyp class prediction probability (%)")
    plt.ylabel(r"IoU$_{clip}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return polyp_preds


def evaluate_plausibility(model:str, xai_method:str, thresholds:List[float]) -> Tuple[float, List[np.ndarray], List[np.ndarray], List[str], List[float], List[np.ndarray], List[float], List[float]]:
    """Compute the mean IoU and std for each polyp-class test sample of a PolypGen6 dataset.
    
        Args: 
            model: The type of classification model used for generating the polyp predictions (e.g. 'finetuned').
            xai_method: The type of XAI method used for generating the explanations (e.g. 'kernelshap').
            thresholds: The thresholds to test; must be in [0.0, 1.0]. If no thresholds provided, every increment of 0.05 will be tested.
        Returns:
            best_t: The best threshold across validation samples that was used to compute IoU score for test samples.
            val_mean_iou_curve: The mean IoU score (across validation samples) for all thresholds, in shape (n_thresholds,).
            val_std_iou_curve: The standard deviation (across validation samples) for all thresholds, in shape (n_thresholds,).
            video_ids: The subclip IDs of each test sample, in shape (n_samples,).
            polyp_preds: The polyp class prediction probability of each test sample, in shape (n_samples,).
            ious_per_frame: The IoU scores for each frame of each test sample, in shape (n_samples, n_frames).
            iou_means: The mean IoU (across frames) of each test sample, in shape (n_samples, n_frames).
            iou_stds: The standard deviation of each test sample's IoU score, in shape (n_samples, n_frames).
    """
    # Get the best threshold of the validation dataset
    best_t, thresholds, val_mean_iou_curve, val_std_iou_curve, all_sample_ious = find_best_threshold(model=model, xai_method=xai_method, thresholds=thresholds)
    # Evaluate threshold dependency
    evaluate_threshold_dependency(
        sample_iou_curves=all_sample_ious,
        mean_iou_curve=val_mean_iou_curve,
        std_iou_curve=val_std_iou_curve,
        thresholds=thresholds,
        save_path=f"./evaluation/plausibility_evaluation_results/threshold_dependency/polypGen6_val_{model}_{xai_method}.png",
    )
    # Get the masks and saliency maps for test dataset
    masks, saliency_maps, video_ids = get_masks_and_saliency_maps(split="test", model=model, xai_method=xai_method)
    # Get IoU mean and std for all samples of the test dataset, using the chosen threshold
    iou_means = []
    iou_stds = []
    ious_per_frame = []   # contains nan values for frames without a polyp
    for sample_idx in range(len(masks)):
        iou_mean, iou_std, iou_per_frame = get_plausibility_score(saliency_map=saliency_maps[sample_idx], mask=masks[sample_idx], threshold=best_t)
        iou_means.append(iou_mean)
        iou_stds.append(iou_std)
        ious_per_frame.append(iou_per_frame)
    # Evaluate frame dependency
    evaluate_frame_dependency(
        ious_per_frame=ious_per_frame, 
        save_path=f"./evaluation/plausibility_evaluation_results/frame_dependency/polypGen6_test_{model}_{xai_method}.png",
    )
    # Evaluate relation to polyp prediction probability
    polyp_preds = evaluate_polyp_prob_relation(
        video_ids=video_ids, 
        iou_means=iou_means, 
        meta_path=f"./xai_methods/saliency_maps/{model}_polypGen6/test/test_ds_metadata.csv", 
        save_path=f"./evaluation/plausibility_evaluation_results/polyp_prop_relation/polypGen6_test_{model}_{xai_method}.png",
    )
    return best_t, val_mean_iou_curve, val_std_iou_curve, video_ids, polyp_preds, ious_per_frame, iou_means, iou_stds