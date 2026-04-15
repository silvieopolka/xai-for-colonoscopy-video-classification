"""Evaluate plausibility of explanations for all dataset-model-settings."""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from typing import List

from data.process_polypgen_data import load_dataset_split
from evaluation.eval_explanation_plausibility import evaluate_plausibility, get_plausibility_score


VALID_CONFIGS = [
    ("polypGen6", "val",  "pretrained"),
    ("polypGen6", "val",  "finetuned"),
    ("polypGen6", "test", "pretrained"),
    ("polypGen6", "test", "finetuned"),
]


def get_all_data_for_eval(thresholds:np.ndarray, polyp_class_only:bool=True) -> None:
    """Create a .csv file that stores all relevant data for each sample of all dataset-model settings. 

        Args:
            thresholds: The thresholds to test, in shape (n_thresholds,).
            polyp_class_only: Whether to skip healthy-class samples.
    """
    # Prepare the csv file
    output_csv = "./evaluation/plausibility_evaluation_results/plausibility_scores.csv"
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_csv):
        raise FileExistsError(f"File at '{output_csv}' already exists.")
    else:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            header = ["dataset", "split", "model", "xai_method", "label", "video_id", "polyp_pred", "n_polyp_frames"] + [f"t={t:.1f}" for t in thresholds]
            writer.writerow(header)

    # Store the plausibility scores
    for dataset, split, model in VALID_CONFIGS:
        ds = load_dataset_split(dataset, split, load_masks=True)
        n_samples = ds.__len__()

        for xai_method in ["kernelshap", "gradcam"]:
            for sample_idx in range(n_samples):
                input_sample, label, video_id, mask = ds[sample_idx]

                if polyp_class_only and (int(label) == 0):
                    continue

                base_path = f"./xai_methods/saliency_maps/{model}_{dataset}/{split}"
                metadata = pd.read_csv(f"{base_path}/{split}_ds_metadata.csv", sep=",")
                polyp_pred = metadata.loc[metadata["video_id"]==video_id, "prob_polyp_percent"].values[0]
                saliency_map = np.load(f"{base_path}/{video_id}/polyp/attr_{xai_method}.npy")
                saliency_map = np.squeeze(saliency_map, axis=-1)

                n_polyp_frames = sum(m.sum() > 0 for m in mask)
                row = [dataset, split, model, xai_method, int(label), video_id, polyp_pred, n_polyp_frames]

                for threshold in thresholds:
                    mean_iou, std_iou, _ = get_plausibility_score(saliency_map, mask, threshold)
                    row.append([mean_iou, std_iou])
                
                with open(output_csv, "a", newline="") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(row)


def build_df(video_ids:List[str], polyp_preds:List[float], iou_means:List[float], iou_stds:List[float], model:str, xai_method:str) -> pd.DataFrame:
    """Build a DataFrame that stores relevant information for a specific dataset-model-setting.
    
        Args:
            video_ids: The subclip IDs of all dataset samples, in shape (n_samples,).
            polyp_preds: The polyp class prediction probabilities of all dataset samples, in shape (n_samples,).
            iou_means: The mean IoU score (across frames) of all dataset samples, in shape (n_samples,).
            iou_stds: The standard deviation of the IoU score of all dataset samples, in shape (n_samples,).
            model: The type of classification model (e.g. 'finetuned').
            xai_method: The type of explainability method (e.g. 'kernelshap').
        Returns:
            The DataFrame.
    """
    return pd.DataFrame({
        "video_id": video_ids,
        "polyp_pred": polyp_preds,
        "mean_iou": iou_means,
        "iou_std": iou_stds,
        "model": model,
        "explainability method": xai_method,
    })


def evaluate_all_threshold_dependencies(mean_curves:dict, thresholds:np.ndarray, save_path:str) -> None:
    """For each dataset-model setting, plot the mean IoU score (across validation samples) over all thresholds.
    
        Args:
            mean_curves: A dictionary containing the mean IoU score (across validation samples) over all thresholds, for each dataset-model setting.
            thresholds: The thresholds to test, in shape (n_thresholds,).
            save_path: Output path for the plot.
    """
    # Prepare the data
    data = []
    for label, curve in mean_curves.items():
        model, xai = label.split("_")
        for t, val in zip(thresholds, curve):
            data.append({
                "threshold": t,
                "IoU_clip": val,
                "model": model,
                "explainability method": xai
            })
    df = pd.DataFrame(data)

    # Draw the lines
    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=df,
        x="threshold",
        y="IoU_clip",
        hue="explainability method",
        style="model",
        linewidth=2.2,
        palette={
            "KernelSHAP": "#4daf4a",   # green
            "Grad-CAM": "#984ea3",     # magenta
        }
    )

    # Add descriptions
    plt.ylim(bottom=0)
    plt.ylabel(r"IoU$_{clip}$")
    plt.title("Threshold Dependency of Explanation Plausibility")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_all_polyp_pred_relations(df:pd.DataFrame, save_path:str) -> None:
    """For each dataset-model setting, \
    (1) plot the mean IoU score (across frames) of each test sample against the sample's polyp class prediction probability \
    and (2) compute the Spearman's rank correlation coefficient.

        Args:
            df: A DataFrame containing, for each dataset-moodel setting, \
                the polyp propabilities and mean IoU scores of all test samples,\
                as well as the model and explainability method.
            save_path: Output path for the plot.
    """
    # Compute spearman correlation
    groups = [
        ("pretrained", "KernelSHAP"),
        ("pretrained", "Grad-CAM"),
        ("finetuned", "KernelSHAP"),
        ("finetuned", "Grad-CAM"),
    ]
    corr_results = {}
    for model, xai in groups:
        subset = df[(df["model"] == model) & (df["explainability method"] == xai)]
        rho, pval = spearmanr(subset["polyp_pred"], subset["mean_iou"])
        corr_results[(model, xai)] = (rho, pval)
    
    # Plot results
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="polyp_pred",
        y="mean_iou",
        hue="explainability method",
        style="model",
        s=80,
        palette={
            "KernelSHAP": "#4daf4a",   # green
            "Grad-CAM": "#984ea3",     # magenta
        },
    )

    # Add descriptions
    text = r"$\bf{Spearman's\ correlation}$"
    for (model, xai), (rho, pval) in corr_results.items():
        text += f"\n{model} + {xai}: $\\rho$={rho:.2f}, p={pval:.3f}"
    plt.text(
        0.25, 0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    plt.title("Relationship between Model Confidence and Explanation Plausibility")
    plt.xlabel("polyp class prediction probability (%)")
    plt.ylabel(r"IoU$_{clip}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_distribution_shift(df:pd.DataFrame, save_path:str) -> None:
    """Create a split violin plot showing the distribution of mean IoU for \
    KernelSHAP and Grad-CAM, split by pretrained vs finetuned models.

        Args:
            df: A DataFrame containing, for each dataset-moodel setting, \
                the mean IoU scores of all test samples,\
                as well as the model and explainability method.
            save_path: Output path for the plot.
    """
    # Plot results
    plt.figure(figsize=(9, 6))
    sns.violinplot(
        data=df,
        x="explainability method",
        y="mean_iou",
        hue="model",
        split=True,
        inner="quartile",
        linewidth=1.2,
        palette={
            "pretrained": "#02daa0",   # purple
            "finetuned": "#ff7f0e",    # orange
        },
        cut=0,
    )

    # Add descriptions
    plt.title("Explanation Plausibility under Data Distribution Shift")
    plt.xlabel("explainability method")
    plt.ylabel(r"IoU$_{clip}$")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(
        title="model",
        loc="upper left",
        frameon=True
    )
    plt.tight_layout()

    # Save plot
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_all_frame_dependencies(df:pd.DataFrame, save_path=str) -> None:
    """Create a split violin plot showing the distribution of IoU standard deviation for\
    pretrained and finetuned models, split by KernelSHAP vs Grad-CAM.

        Args:
            df: A DataFrame containing, for each dataset-moodel setting, \
                the mean IoU scores of all test samples,\
                as well as the model and explainability method.
            save_path: Output path for the plot.
    """
    # Plot results
    plt.figure(figsize=(9, 6))
    sns.violinplot(
        data=df,
        x="model",
        y="iou_std",
        hue="explainability method",
        split=True,
        inner="quartile",
        linewidth=1.2,
        palette={
            "KernelSHAP": "#4daf4a",   # green
            "Grad-CAM": "#984ea3",     # magenta
        },
        cut=0,
    )

    # Add descriptions
    plt.title("Consistency of Explanation Plausibility across Frames")
    plt.xlabel("model")
    plt.ylabel(r"standard deviation of IoU$_{clip}$")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(
        title="explainability method",
        loc="upper left",
        frameon=True
    )
    plt.tight_layout()

    # Save plot
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    thresholds = np.linspace(0, 1, 11)

    # Generate a .csv file that lists all raw information at once
    get_all_data_for_eval(thresholds=thresholds)

    # Evaluate plausibility across all dataset-model settings
    pKS_best_t, pKS_val_mean_iou_curve, pKS_val_std_iou_curve, pKS_video_ids, pKS_polyp_preds, pKS_ious_per_frame, pKS_iou_means, pKS_iou_stds = evaluate_plausibility(model="pretrained", xai_method="kernelshap", thresholds=thresholds)
    pGC_best_t, pGC_val_mean_iou_curve, pGC_val_std_iou_curve, pGC_video_ids, pGC_polyp_preds, pGC_ious_per_frame, pGC_iou_means, pGC_iou_stds = evaluate_plausibility(model="pretrained", xai_method="gradcam", thresholds=thresholds)
    fKS_best_t, fKS_val_mean_iou_curve, fKS_val_std_iou_curve, fKS_video_ids, fKS_polyp_preds, fKS_ious_per_frame, fKS_iou_means, fKS_iou_stds = evaluate_plausibility(model="finetuned", xai_method="kernelshap", thresholds=thresholds)
    fGC_best_t, fGC_val_mean_iou_curve, fGC_val_std_iou_curve, fGC_video_ids, fGC_polyp_preds, fGC_ious_per_frame, fGC_iou_means, fGC_iou_stds = evaluate_plausibility(model="finetuned", xai_method="gradcam", thresholds=thresholds)

    # Plot the threshold dependency for the validation data
    val_mean_iou_curves = {
        "pretrained_KernelSHAP": pKS_val_mean_iou_curve,
        "pretrained_Grad-CAM":   pGC_val_mean_iou_curve,
        "finetuned_KernelSHAP":  fKS_val_mean_iou_curve,
        "finetuned_Grad-CAM":    fGC_val_mean_iou_curve,
    }

    evaluate_all_threshold_dependencies(
        mean_curves=val_mean_iou_curves, 
        thresholds=thresholds,
        save_path="./evaluation/plausibility_evaluation_results/threshold_dependency/polypGen6_val_combined.png",
    )

    # Build a data frame for the PolypGen6 test dataset results
    df = pd.concat([
        build_df(pKS_video_ids, pKS_polyp_preds, pKS_iou_means, pKS_iou_stds, "pretrained", "KernelSHAP"),
        build_df(pGC_video_ids, pGC_polyp_preds, pGC_iou_means, pGC_iou_stds, "pretrained", "Grad-CAM"),
        build_df(fKS_video_ids, fKS_polyp_preds, fKS_iou_means, fKS_iou_stds, "finetuned", "KernelSHAP"),
        build_df(fGC_video_ids, fGC_polyp_preds, fGC_iou_means, fGC_iou_stds, "finetuned", "Grad-CAM"),
    ], ignore_index=True)

    # Plot relationship between model confidence and explanation plausibility
    evaluate_all_polyp_pred_relations(
        df=df,
        save_path="./evaluation/plausibility_evaluation_results/polyp_prop_relation/polypGen6_test_combined.png",
    )

    # Plot explanation plausibility under data distribution shift
    evaluate_distribution_shift(
        df=df,
        save_path="./evaluation/plausibility_evaluation_results/distribution_shift/polypGen6_test_combined.png",
    )

    # Plot consistency of explanation plausibility across frames
    evaluate_all_frame_dependencies(
        df=df,
        save_path="./evaluation/plausibility_evaluation_results/frame_dependency/polypGen6_test_combined.png",
    )