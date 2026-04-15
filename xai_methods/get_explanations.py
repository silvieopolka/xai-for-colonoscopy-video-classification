"""This scripts generates Grad-CAM and KernelSHAP explanations for EndoFM-LV prediction of PolypGen test sample(s)."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import csv
import logging
import numpy as np
import time
import torch

from data.polypGen6.config import CfgPolypGen6
from data.process_polypgen_data import PolypGenDataset
from model.helpers import pick_device
from model.model_loading import load_classification_model
from xai_methods.apply_GradCAM import compute_gradcam_attribution
from xai_methods.apply_KernelSHAP import (
    apply_slic_per_frame, 
    apply_slic_per_video, 
    compute_kernelshap_attribution,
    get_grid_segments,
)
from xai_methods.visualization import visualize_video_saliency


def format_runtime(seconds:float) -> str:
    """Get the runtime of an XAI method in a readable format.""" 
    h = int(seconds // 3600) 
    m = int((seconds % 3600) // 60) 
    s = int(seconds % 60) 
    return f"{h}h {m}min {s}sec"


def get_explanation_for_input_sample(
    args:argparse.Namespace, 
    ds:torch.utils.data.Dataset, 
    input_sample_idx:int, 
    device:torch.device,
    result_dir:str,
) -> dict:
    """Generate Grad-CAM and KernelSHAP explanations for a single PolypGen test sample."""
    # Get the PolypGen input sample
    input_sample, label, video_id, _ = ds.__getitem__(idx=input_sample_idx)

    # Since the model expects a batch dimension and we only pass 1 input sample, we add one
    input_sample = input_sample.unsqueeze(0)
    input_sample = input_sample.to(device, dtype=torch.float32)
    logging.info(f"Video tensor shape: {input_sample.shape}")  # The model (PatchEmbed) expects the shape [B, C, T, H, W]
    logging.info(f"Label: {label}")

    # Get the data configuration
    cfg = CfgPolypGen6()

    # Get the model
    model = load_classification_model(cfg=cfg, model=args.model)
    model = model.to(device, dtype=torch.float32)

    # Generate the prediction
    model.eval()
    with torch.no_grad():
        features = model(input_sample)[0]  # get the cls token of shape [B, 768]
        outputs = model.head(features)     # shape [B, 2]
        # get prediction
        prediction = outputs.argmax(dim=1).item()
        # get probabilities
        probs = torch.softmax(outputs, dim=1)
        prob_healthy = probs[0, 0].item() * 100 
        prob_polyp = probs[0, 1].item() * 100
        logging.info(f"Probabilities: {prob_healthy:.2f}% Class 0 (healthy) and {prob_polyp:.2f}% Class 1 (polyp)")
        logging.info(f"Prediction: {prediction}")
        logging.info(f"Ground Truth: {label}")

    # Choose for which class to generate the explanation
    if args.class_of_interest == "polyp":
        target_class = 1
    elif args.class_of_interest == "healthy":
        target_class = 0
    elif args.class_of_interest == "prediction":
        target_class = prediction
    elif args.class_of_interest == "label":
        target_class = label
    else:
        raise ValueError(f"{args.class_of_interest} is not a valid class for generating the explanation.")
    
    # Set the directory to save the saliency maps
    save_dir = f"{result_dir}/{video_id}/{args.class_of_interest}" 
    os.makedirs(save_dir, exist_ok=True)

    # Run Grad-CAM
    start = time.perf_counter()

    raw_attr_gradcam = compute_gradcam_attribution(
        model=model, 
        input_sample=input_sample, 
        target_class=target_class,
    )

    end = time.perf_counter()
    elapsed_seconds = (end - start)
    gradcam_runtime = format_runtime(seconds=elapsed_seconds)
    logging.info(f"Grad-CAM took {gradcam_runtime}")
    logging.info(f"Shape of Grad-CAM's saliency map: {raw_attr_gradcam.shape}")

    # Run KernelSHAP
    start = time.perf_counter()
    n_frames = input_sample.shape[2]

    if args.feature_division == "SLIC_frame":
        n_segments = 15   # per frame
        segments_per_frame = apply_slic_per_frame(input_sample, n_segments=n_segments, compactness=15)
        actual_n_segments_per_frame = [len(np.unique(seg)) for seg in segments_per_frame]
        if not all(actual_n_segments_per_frame[frame_idx] == n_segments for frame_idx in range(n_frames)):
            logging.warning(f"SLIC did not produce the requested number of segments for all frames.")
            logging.info(f"Actual number of segments per frame: {actual_n_segments_per_frame}")
        n_input_features = sum(actual_n_segments_per_frame)
    
    elif args.feature_division == "SLIC_clip":
        n_segments = 17   # per clip
        segments_per_frame = apply_slic_per_video(input_sample, n_segments=n_segments, compactness=20)
        actual_n_segments = len(np.unique(segments_per_frame))
        if actual_n_segments != n_segments:
            logging.warning(f"SLIC did not produce the requested number of segments for the clip.")
            logging.info(f"Actual number of input features: {actual_n_segments}")
        n_input_features = actual_n_segments
    
    elif args.feature_division == "grid":
        n_segments = [14, 14]   # resolution per img axis (choose 14 to get identical resolution as Grad-CAM explanation)
        segments_per_frame = get_grid_segments(input_sample, n_segments)
        n_input_features = n_frames * n_segments[0] * n_segments[1]
    
    else:
        raise ValueError(f"{args.feature_division} is not a valid method for defining input features for KernelSHAP.")
    
    logging.info(f"KernelSHAP will be run with {n_input_features} input features.")
    raw_attr_kernelshap = compute_kernelshap_attribution(
        model=model, 
        input_sample=input_sample, 
        target_class=target_class,  
        n_input_features=n_input_features,
        segments=segments_per_frame,
        device=device,
    )

    end = time.perf_counter() 
    elapsed_seconds = (end - start)
    kernelshap_runtime = format_runtime(seconds=elapsed_seconds)
    logging.info(f"KernelSHAP took {kernelshap_runtime}")
    logging.info(f"Shape of KernelShap's saliency map: {raw_attr_kernelshap.shape}")

    # Visualize the explanations
    visualize_video_saliency(
        video_tensor=input_sample, 
        raw_attr_gradcam=raw_attr_gradcam, 
        raw_attr_kernelshap=raw_attr_kernelshap, 
        sample_id=video_id, 
        model_name=args.model,
        segments_per_frame=segments_per_frame,
        feature_division=args.feature_division,
        n_segments=n_segments,
        save_dir=save_dir,
        add_boundaries=args.add_boundaries,
        interpolate=args.interpolate,
    )

    # Collect metadata for logging
    metadata = {
        "video_id": video_id, 
        "label": int(label), 
        "prediction": int(prediction), 
        "prob_healthy_percent": round(prob_healthy, 2), 
        "prob_polyp_percent": round(prob_polyp, 2), 
        "gradcam_runtime": gradcam_runtime, 
        "kernelshap_runtime": kernelshap_runtime, 
        "kernel_shap_feature_division": args.feature_division, 
        "kernel_shap_n_input_features": n_input_features,
    }

    return metadata


def add_input_sample_metadata(result_dir:str, row_dict:dict, dataset:str):
    """Store the metadata of the explanation generation process for an input sample."""
    csv_path = f"{result_dir}/{dataset}_ds_metadata.csv" 
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f: 
        writer = csv.DictWriter(f, fieldnames=row_dict.keys()) 
        if not file_exists: 
            writer.writeheader() 
        writer.writerow(row_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get explanations for a PolypGen6 clip prediction.')
    parser.add_argument('--split', type=str, choices=["train", "val", "test"], help="The dataset split to use for explanation generation.")
    parser.add_argument('--all_test_samples', action="store_true", help="Whether to run the explanation generation for all PolypGen samples of the dataset split, or for a specified one (args.row_idx).")
    parser.add_argument('--device', type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="The device to use for KernelSHAP calculation.")
    parser.add_argument('--model', type=str, default="pretrained", choices=["pretrained", "finetuned"], help="The model for generating the prediction. 'pretrained' refers to the PolypDiag-finetuned EndoFM-LV model, while 'finetuned' refers to the pretrained model that we finetuned on PolypGen data.")
    parser.add_argument('--row_idx', type=int, default=0, help="The row index of the PolypGen sample.")
    parser.add_argument('--class_of_interest', type=str, default="polyp", choices=["polyp", "healthy", "prediction", "label"], help="The class for which the explanation is generated.")
    parser.add_argument('--feature_division', type=str, default="SLIC_clip", choices=["grid", "SLIC_frame", "SLIC_clip"], help="The method used to define input features for KernelSHAP.")
    parser.add_argument('--add_boundaries', action="store_true", help="Add superpixel boundaries in the visualization.")
    parser.add_argument('--interpolate', action="store_true", help="Interpolate the saliency values for visualization convenience.")
    args = parser.parse_args()

    # Enable logging messages
    logging.basicConfig(level=logging.INFO)

    # Set the device
    device = pick_device(args.device)
    logging.info(f"Using device {device}.")

    # Get the PolypGen dataset
    ds = PolypGenDataset(
        split=args.split,
        num_frames=6,
        data_root="./data/polypGen6/data",
        masks_root="./data/polypGen6/mask",
        meta_path="./data/polypGen6/meta_data.csv",
    )
    n_input_samples = ds.__len__()

    # Generate the Grad-CAM and KernelSHAP explanations
    result_dir = f"./xai_methods/saliency_maps/{args.model}_polypGen6/{args.split}"
    if args.all_test_samples:
        logging.getLogger().setLevel(logging.WARNING)
        # Conduct the explanation generation for all test samples
        for idx in range(n_input_samples):
            print(f"Generating explanations for sample {idx} of {n_input_samples}")
            meta_data = get_explanation_for_input_sample(
                args=args, 
                ds=ds, 
                input_sample_idx=idx, 
                device=device,
                result_dir=result_dir,
            )
            # Collect the metadata
            add_input_sample_metadata(result_dir, meta_data, dataset=args.split)
    else:
        # Conduct the explanation generation for the specified input sample
        _ = get_explanation_for_input_sample(
            args=args, 
            ds=ds, 
            input_sample_idx=args.row_idx,
            device=device,
            result_dir=result_dir,
        )