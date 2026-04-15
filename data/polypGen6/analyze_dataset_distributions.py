"""Plot data distributions of the 6-frame PolypGen dataset."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def plot_class_distribution(df:pd.DataFrame, result_dir:str) -> None:
    """Get the class and hospital distribution of the 6-frame PolypGen training, validation and test datasets.
    
        Args:
            df: The meta_data.csv containing information of the PolypGen6 dataset.
            result_dir: Output path for the plot.
    """
    df_plot = df.copy()
    # Count healthy/polyp samples per dataset and hospital
    df_plot["class_name"] = df_plot["label"].map({0: "healthy", 1: "polyp"})
    counts = (
        df_plot
        .groupby(["dataset", "class_name", "dataCollectionCenter"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot_table(
        index=["dataset", "class_name"],
        columns="dataCollectionCenter",
        values="count",
        fill_value=0
    )
    hospitals = ["Oslo University Hospital", "University of Alexandria"]
    pivot = pivot[hospitals]
    datasets = ["train", "val", "test"]
    classes = ["healthy", "polyp"]
    # Build x-axis positions and corresponding values for plotting
    x_positions = []
    x_labels = []
    pos = 0
    for ds in datasets:
        for cls in classes:
            x_positions.append(pos)
            x_labels.append(cls)
            pos += 1
    oslo_vals = []
    alex_vals = []
    for ds in datasets:
        for cls in classes:
            if (ds, cls) in pivot.index:
                oslo_vals.append(pivot.loc[(ds, cls), "Oslo University Hospital"])
                alex_vals.append(pivot.loc[(ds, cls), "University of Alexandria"])
            else:
                oslo_vals.append(0)
                alex_vals.append(0)
    # Plot figure
    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, oslo_vals, label="Oslo University Hospital", color="#4C72B0")
    plt.bar(x_positions, alex_vals, bottom=oslo_vals, label="University of Alexandria", color="#55A868")
    plt.xticks(x_positions, x_labels)
    for i, ds in enumerate(datasets):
        center = i * 2 + 0.5
        plt.text(center, -max(oslo_vals + alex_vals) * 0.05, ds,
                 ha="center", va="top", fontsize=12)
    plt.ylabel("Number of Input Samples")
    plt.title("Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{result_dir}/class_dist.png", dpi=200)
    plt.close()


def extract_seq_number(seq:str) -> int:
    """Extract the sequence number from the name of a PolypGen data sample.
    
        Args:
            seq: The subclip name of a PolypGen6 sample (e.g. 'seq1_clip1').
        Returns:
            seq_number: The sequence number (e.g. 1).
    """
    match = re.match(r"seq(\d+)", seq)
    if match:
        seq_number = int(match.group(1))
    else:
        raise ValueError(f"Sequence number cannot be extracted from '{seq}'.")
    return seq_number


def plot_subclips_per_sequence(df:pd.DataFrame, result_dir:str) -> None:
    """Get the number of subclips per original sequence.
    
        Args:
            df: The meta_data.csv containing information of the PolypGen6 dataset.
            result_dir: Output path for the plot.
    """
    # Get original video id
    df["sequence"] = df["video_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["class_name"] = df["label"].map({0: "healthy", 1: "polyp"})
    # Enforce ordering
    df["dataset"] = pd.Categorical(df["dataset"], categories=["train", "val", "test"], ordered=True)
    df["class_name"] = pd.Categorical(df["class_name"], categories=["healthy", "polyp"], ordered=True)
    # Count subclips per video
    seq_counts = (
        df.groupby("sequence")
          .agg(
              count=("video_id", "size"),
              dataset=("dataset", "first"),
              class_name=("class_name", "first")
          )
          .reset_index()
    )
    # Sort sequences: dataset → class → video_id
    seq_counts["seq_num"] = seq_counts["sequence"].apply(extract_seq_number)
    seq_counts = seq_counts.sort_values(["dataset", "class_name", "seq_num"])
    ordered_sequences = seq_counts["sequence"].tolist()
    # Plot figure
    plt.figure(figsize=(18, 6))
    sns.barplot(
        data=seq_counts,
        x="sequence",
        y="count",
        hue="dataset",
        palette="Set1",
        order=ordered_sequences,
        errorbar=None,
        dodge=False,
    )
    plt.ylabel("Number of Subclips")
    plt.xlabel("Colonoscopic Video ID")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/subclip_dist.png", dpi=200)
    plt.close()


def compute_polyp_size(mask_dir:str) -> float:
    """Compute the average polyp pixel ratio across the 6 frames of a PolypGen data sample.
    
        Args:
            mask_dir: Path to the PolypGen6 sample's segmentation masks.
        Returns:
            The average polyp pixel ratio (across frames).
    """
    sizes = []
    for i in range(1, 7):
        mask_path = os.path.join(mask_dir, f"{i}.jpg")
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        white = (mask > 127).sum()
        total = mask.size
        sizes.append(white / total)
    return np.mean(sizes) if sizes else np.nan


def plot_polyp_size(df:pd.DataFrame, mask_base_dir:str, result_dir:str) -> None:
    """Get the polyp size distribution of the 6-frame PolypGen training, validation and test datasets.
    
        Args:
            df: The meta_data.csv containing information of the PolypGen6 dataset.
            mask_base_dir: The path to the PolypGen6 'mask' folder.
            result_dir: Output path for the plot.
    """
    df_polyp = df[df["label"].astype(int) == 1].reset_index(drop=True)
    # Get the average polyp size of all data samples
    sizes = []
    for _, row in df_polyp.iterrows():
        masks = os.path.join(mask_base_dir, row["dataset"], row["video_id"])
        size = compute_polyp_size(masks)
        sizes.append(size)
    df_polyp["polyp_size"] = sizes
    # Plot figure
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df_polyp,
        x="dataset",
        y="polyp_size",
        palette="Set3"
    )
    sns.stripplot(
        data=df_polyp,
        x="dataset",
        y="polyp_size",
        color="black",
        alpha=0.5,
        jitter=False,
        dodge=False,
    )
    plt.title("Polyp Size Distribution per Dataset")
    plt.ylabel("Average Polyp Pixel Ratio")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/polyp_size_dist.png", dpi=200)
    plt.close()


if __name__ == '__main__':
    meta_path = "./data/polypGen6/meta_data.csv"
    mask_dir = "./data/polypGen6/mask"
    result_dir = "./data/polypGen6/ds_distribution_info"

    df = pd.read_csv(meta_path, sep=";")

    plot_class_distribution(df, result_dir)
    plot_subclips_per_sequence(df, result_dir)
    plot_polyp_size(df, mask_dir, result_dir)