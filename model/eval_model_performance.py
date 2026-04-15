"""A collection of functions to evaluate the classification model's performance."""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve, 
    average_precision_score,
)


def get_predictions(
    model:torch.nn.Module, 
    ds_loader:torch.utils.data.DataLoader, 
    device:torch.device, 
    model_architecture:str='endofm_lv',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions for a given dataset."""
    model.eval()
    all_probs, all_preds, all_targets, ids = [], [], [], []

    with torch.no_grad():
        for inputs, targets, video_ids, _ in ds_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if model_architecture == 'endofm_lv':
                features = model(inputs)[0]
                outputs = model.head(features)
            else:
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]   # the probability of class 1 (polyp); the probabilities of class 1 and 2 sum up to 1.0 
            preds = outputs.argmax(dim=1)
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            ids.extend(video_ids)
    
    return np.array(all_probs), np.array(all_preds), np.array(all_targets), np.array(video_ids)


def plot_confusion_matrix(targets:np.ndarray, predictions:np.ndarray, save_dir:str, details:str=""):
    """Generate the confusion matrix."""
    cm = confusion_matrix(targets, predictions)

    # get the highest number of test samples in any class
    class_totals = cm.sum(axis=1) 
    vmax = class_totals.max() 

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Healthy', 'Polyp'], yticklabels=['Healthy', 'Polyp'],
                annot_kws={'size':14}, vmin=0, vmax=vmax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f"Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_dir = os.path.join(save_dir, f"{details}confusion_matrix.png")
    plt.savefig(cm_dir, bbox_inches='tight')
    plt.close()


def plot_roc_curve(targets:np.ndarray, probabilities:np.ndarray, save_dir:str, details:str=""):
    """Generate the ROC curve."""
    fpr, tpr, _ = roc_curve(targets, probabilities)
    auc_score = roc_auc_score(targets, probabilities)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_dir = os.path.join(save_dir, f"{details}roc_curve.png")
    plt.savefig(roc_dir, bbox_inches='tight')
    plt.close()


def plot_pr_curve(all_targets:np.ndarray, all_probs:np.ndarray, save_dir:str, details:str=""):
    """Generate the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    ap_score = average_precision_score(all_targets, all_probs)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap_score:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_dir = os.path.join(save_dir, f"{details}precision_recall_curve.png")
    plt.savefig(pr_dir, bbox_inches='tight')
    plt.close()


def evaluate_model_performance(
    model:torch.nn.Module, 
    test_loader:torch.utils.data.DataLoader, 
    device:torch.device, 
    save_dir:str, 
    model_architecture:str='endofm_lv',
    details:str="", 
):
    """Evaluate model on the test dataset.
    
    Get accuracy, F1 score, precision, recall.
    Plot confusion matrix, ROC curve, and PR curve.
    """
    # Get the predictions for all test samples
    all_probs, all_preds, all_targets, all_video_ids = get_predictions(
        model=model,
        ds_loader=test_loader,
        device=device,
        model_architecture=model_architecture,
    )

    # Get basic metric scores
    acc = ((torch.tensor(all_preds) > 0.5) == torch.tensor(all_targets)).float().mean().item()
    f1 = f1_score(all_targets, all_preds) 
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)

    # Plot the confusion matrix
    plot_confusion_matrix(all_targets, all_preds, save_dir, details)

    if len(set(all_targets)) > 1:
        # Plot the ROC curve
        plot_roc_curve(all_targets, all_probs, save_dir, details)
        # Plot the Precision-Recall curve
        plot_pr_curve(all_targets, all_probs, save_dir, details)
    
    return all_video_ids, all_targets, all_probs, acc, f1, precision, recall