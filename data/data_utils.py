"""A collection of functions to manipulate datasets."""

import logging
import random

from torch.utils.data import Dataset, Subset


# NOTE: All PolypGen6 dataset splits are already perfectly balanced
def get_balanced_subset(dataset:Dataset, per_class:int) -> Subset:
    """Randomly sample an equal number of data points of each class.
    
        Args:
            dataset: The dataset to sample from. It must comprise exactly two classes (0 and 1).
            per_class: The number of data points to sample per class.
        
        Returns:
            balanced_dataset: The balanced subset.
    """
    labels = dataset.labels
    idx_class0 = labels[labels["label"] == 0].index.tolist()
    idx_class1 = labels[labels["label"] == 1].index.tolist()

    chosen0 = random.sample(idx_class0, min(per_class, len(idx_class0)))
    chosen1 = random.sample(idx_class1, min(per_class, len(idx_class1)))

    if len(idx_class0) < per_class:
        logging.info(f"The dataset does not contain the requested amount of samples of class 0. Only {len(idx_class0)} were selected.")
    if len(idx_class1) < per_class:
        logging.info(f"The dataset does not contain the requested amount of samples of class 1. Only {len(idx_class1)} were selected.")

    balanced_dataset = Subset(dataset, chosen0 + chosen1)

    return balanced_dataset