"""This script trains a binary classification model on PolypGen6 data."""

import argparse
import csv
import logging
import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
)

from data.data_utils import get_balanced_subset
from data.polypGen6.config import CfgPolypGen6
from data.process_polypgen_data import get_polypgen_datasets, collate_polypgen_batch
from model.eval_model_performance import evaluate_model_performance
from model.alternative_architectures import get_architecture
from model.helpers import pick_device
from model.model_loading import load_backbone_model, load_classification_model
from xai_methods.apply_GradCAM import compute_gradcam_attribution


BASE_SEED = 0
def seed_worker(worker_id:int): 
    worker_seed = BASE_SEED + worker_id 
    random.seed(worker_seed) 
    np.random.seed(worker_seed) 
    torch.manual_seed(worker_seed) 


def sanity_checks(args:argparse.Namespace, train_loader:DataLoader, device:torch.device, backbone:torch.nn.Module=None):
    """Check whether the current setup matches expectations."""
    sample_batch = next(iter(train_loader))
    expected_input_shape = (args.batch_size, 3, args.num_frames, 224, 224)   # [B, C, T, H, W]
    expected_label_shape = [(args.batch_size,), (args.batch_size, 1)]
    if sample_batch[0].shape != expected_input_shape:
        raise ValueError(f"The input shape ({sample_batch[0].shape}) does not match the expected shape ({expected_input_shape}).")
    if sample_batch[1].shape not in expected_label_shape:
        raise ValueError(f"The label shape ({sample_batch[1].shape}) does not match the expected shape ({expected_label_shape[0]} or {expected_label_shape[1]}).")

    if backbone is not None:
        with torch.no_grad():
            sample_out = backbone(sample_batch[0].to(device))[0]
        expected_backbone_output_shape = (args.batch_size, 768)
        if sample_out.shape != expected_backbone_output_shape:
            raise ValueError(f"The backbone output shape ({sample_out.shape}) does not match the expected shape ({expected_backbone_output_shape}).")


def unfreeze_last_n_blocks(model:torch.nn.Module, n_blocks:int=2, unfreeze_embedding_layer:bool=False):
    """Unfreeze the last n transformer blocks of the backbone for finetuning.
    
    Args:
        model: The EndoFM-LV model (a Vision Timesformer architecture)
        n_blocks: Number of final blocks to unfreeze
        unfreeze_embedding_layer: Whether to allow finetuning the patch embedding layer.
    
    Returns:
        Tuple of (backbone_params, head_params) for optimizer
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # The ViT has 12 blocks total (depth=12)
    # Unfreeze last n blocks
    total_blocks = 12
    start_block = total_blocks - n_blocks
    
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        # Unfreeze last n blocks
        for block_idx in range(start_block, total_blocks):
            if f'blocks.{block_idx}.' in name:
                param.requires_grad = True
                backbone_params.append(param)
                break
        # Unfreeze the patch embedding layer if requested
        if unfreeze_embedding_layer:
            if "patch_embed.proj" in name:
                param.requires_grad = True
        # Always unfreeze the classification head
        if name.startswith('head'):
            param.requires_grad = True
            head_params.append(param)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return backbone_params, head_params


def initialize_model(args:argparse.Namespace, cfg:CfgPolypGen6, device:torch.device, train_loader:DataLoader):
    """Initialize the video classification model, and set the optimizer, scheduler and criterion for finetuning."""
    logging.info(f"Initializing {args.model_architecture.upper()} architecture")

    # ------------------------------------------------------------------------------------------------------
    # EndoFM-LV MODEL
    # ------------------------------------------------------------------------------------------------------
    if args.model_architecture == 'endofm_lv':
        if args.prep_endofm_lv_model == 'finetuned':
            # Load both backbone and classification head weights of the PolypDiag-finetuned EndoFM-LV model
            model = load_classification_model(cfg=cfg, model="pretrained").to(device)
            sanity_checks(args, train_loader, device, backbone=None)
        elif args.prep_endofm_lv_model == 'finetuned_backboneOnly':
            # Only keep the backbone weights of the PolypDiag-finetuned EndoFM-LV model
            model = load_classification_model(cfg=cfg, model="pretrained").to(device)
            model.head = torch.nn.Identity()
            sanity_checks(args, train_loader, device, backbone=model)
            # Add a new binary classification head
            model.head = torch.nn.Linear(768, 2).to(device)
        else:
            # Load the original backbone weights of the EndoFM-LV model
            model = load_backbone_model(cfg=cfg).to(device)
            sanity_checks(args, train_loader, device, backbone=model)
            # Add a new binary classification head
            model.head = torch.nn.Linear(768, 2).to(device)
        
        # Freeze backbone parameters as requested
        if args.unfreeze_endofm_lv_blocks > 0:
            # Unfreeze the last N transformer blocks
            logging.info(f"Unfreezing last {args.unfreeze_endofm_lv_blocks} transformer blocks")
            backbone_params, head_params = unfreeze_last_n_blocks(model, args.unfreeze_endofm_lv_blocks, args.use_saliency_loss)
        else:
            # Freeze all backbone parameters
            logging.info("Freezing all backbone parameters (only training head)")
            for name, param in model.named_parameters(): 
                if not name.startswith("head"): 
                    param.requires_grad = False
                if args.use_saliency_loss:
                    if "patch_embed.proj" in name:
                        param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
            head_params = list(model.head.parameters())
            backbone_params = []
        
        # Set the training parameters with differential learning rates
        if args.unfreeze_endofm_lv_blocks > 0 and backbone_params:
            logging.info(f"Using differential learning rates: head={args.lr}, backbone={args.lr * 0.1}")
            optimizer = AdamW([
                {'params': head_params, 'lr': args.lr},
                {'params': backbone_params, 'lr': args.lr * 0.1}   # do not overwrite pretrained weights as aggressively
            ], weight_decay=1e-4)
        else:
            optimizer = AdamW(head_params, lr=args.lr, weight_decay=1e-4)

    # ------------------------------------------------------------------------------------------------------
    # ALTERNATIVE ARCHITECTURES
    # ------------------------------------------------------------------------------------------------------
    else:
        # Create model instance
        model = get_architecture(args.model_architecture).to(device)
        # All parameters are trainable for alternative architectures
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logging.info(f"The alternative architecture is trained from scratch. All {len(trainable_params)} parameters are trainable.")
        # Single learning rate for all parameters
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Set remaining training parameters
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


def save_finetuned_model(
    model:torch.nn.Module, 
    epoch:int, 
    optimizer:torch.optim.Optimizer, 
    scheduler:torch.optim.lr_scheduler._LRScheduler, 
    amp_enabled:bool, 
    scaler:GradScaler,
    args:argparse.Namespace, 
    best_metric:float,
) -> dict:
    """Build a checkpoint dictionary for saving the finetuned model."""
    
    if args.model_architecture == 'endofm_lv':
        # EndoFM-LV: Save backbone and head separately for compatibility
        backbone_state_dict = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("head")
        }
        head_state_dict = {f"head.{k}": v for k, v in model.head.state_dict().items()}
    else:
        # Alternative architectures: Save full model state
        backbone_state_dict = model.state_dict()
        head_state_dict = {}

    save_dict = {
        "epoch": epoch,
        "model_architecture": args.model_architecture,
        "backbone_state_dict": backbone_state_dict,
        "state_dict": head_state_dict,
        "full_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_enabled,
        "scaler": scaler.state_dict() if amp_enabled and scaler is not None else None,
        "monitor": args.monitor,
        "best_metric": best_metric,
        "args": vars(args),
    }
    return save_dict


def saliency_criterion(inputs, targets, masks, model:torch.nn.Module, device:torch.device):
    """Compute the saliency loss for all polyp-labelled samples."""
    polyp_indices = (targets == 1).nonzero(as_tuple=True)[0]
    saliency_losses = []
    if len(polyp_indices) > 0:
        for idx in polyp_indices:
            sample = inputs[idx:idx+1]  # shape [1, 3, T, H, W]
            mask = masks[idx:idx+1]     # shape [1, T, H, W]
            # Compute raw Grad-CAM saliency map
            raw_attr = compute_gradcam_attribution(model, sample, target_class=1, device=device)
            # Upsample to input resolution
            _, _, sal_h, sal_w = raw_attr.shape
            patch_h = sample.shape[-2] // sal_h
            patch_w = sample.shape[-1] // sal_w
            attr_gradcam = raw_attr.repeat_interleave(patch_h, dim=2).repeat_interleave(patch_w, dim=3)
            # Normalize using absolute values
            attr_gradcam = attr_gradcam / (attr_gradcam.abs().sum() + 1e-8)
            # ------------------------------------------------------------------------------------------
            # Loss component to penalize positive contribution to polyp class in background region
            # ------------------------------------------------------------------------------------------
            p_attr_gradcam = torch.relu(attr_gradcam)
            background = 1 - mask  # flip mask labelling (background: 0 -> 1, polyp: 1 -> 0)
            p_sal_loss = (background * p_attr_gradcam.squeeze(1)).sum()
            # ------------------------------------------------------------------------------------------
            # Loss component to penalize negative contribution to polyp class in polyp region
            # ------------------------------------------------------------------------------------------
            n_attr_gradcam = torch.relu(-attr_gradcam)
            n_sal_loss = (mask * n_attr_gradcam.squeeze(1)).sum()
            # ------------------------------------------------------------------------------------------
            # Saliency loss of polyp-labelled sample
            # ------------------------------------------------------------------------------------------
            sal_loss = (0.5 * p_sal_loss) + (0.5 * n_sal_loss)
            saliency_losses.append(sal_loss)
    return saliency_losses


def finetune_model(
    args:argparse.Namespace, 
    model:torch.nn.Module, 
    train_loader:DataLoader, 
    val_loader:DataLoader, 
    optimizer:torch.optim.Optimizer, 
    amp_enabled:bool, 
    criterion:torch.nn.Module, 
    scaler:GradScaler, 
    scheduler:torch.optim.lr_scheduler._LRScheduler, 
    device:torch.device,
    run_dir:str, 
    metrics_dir:str, 
    best_model_dir:str,
):
    """Finetune the classification model."""
    metrics_records = []
    best_metric = float('inf') if args.monitor == "val_loss" else float('-inf')
    best_epoch = None
    epochs_since_improve = 0
    stopped_early = False

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_probs, train_preds, train_targets = [], [], []
        for inputs, targets, _, all_masks in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(enabled=amp_enabled):
                if args.model_architecture == 'endofm_lv':
                    # EndoFM-LV uses separate backbone and head
                    features = model(inputs)[0]      # backbone output -> get the cls token
                    outputs = model.head(features)   # classification head
                else:
                    # Alternative architectures
                    outputs = model(inputs)
                
                # ------------------------------------------------
                # CLASSIFICATION LOSS
                # ------------------------------------------------
                classification_loss = criterion(outputs, targets.long())

                # ------------------------------------------------
                # SALIENCY LOSS (only for polyp-labelled samples)
                # ------------------------------------------------
                if args.use_saliency_loss:
                    # Convert masks into a tensor of shape [B, T, H, W]
                    masks = []
                    for m in all_masks:
                        if m is None:
                            # healthy sample → all-zero mask
                            masks.append(torch.zeros((args.num_frames, 224, 224), dtype=torch.float32))
                        else:
                            masks.append(torch.from_numpy(m).float())
                    masks = torch.stack(masks, dim=0).to(device)

                    saliency_losses = saliency_criterion(
                        inputs=inputs,
                        targets=targets,
                        masks=masks,
                        model=model,
                        device=device,
                    )
                    if len(saliency_losses) > 0:
                        loss_saliency = torch.stack(saliency_losses).mean()
                        loss = classification_loss + args.alpha * loss_saliency
                    else: 
                        loss = classification_loss
                else:
                    loss = classification_loss

            if amp_enabled:
                scaler.scale(loss).backward()
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    # Clip gradients for all trainable parameters
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm > 0:
                    # Clip gradients for all trainable parameters
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm)
                optimizer.step()

            running_loss += classification_loss.item() * inputs.size(0)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            batch_preds = outputs.argmax(dim=1)   # We use the default classification threshold of 0.5
            correct += (batch_preds.squeeze() == targets.int()).sum().item()
            total += targets.size(0)
            train_probs.extend(batch_probs.tolist())
            train_preds.extend(batch_preds.view(-1).detach().cpu().tolist())
            train_targets.extend(targets.view(-1).detach().cpu().tolist())
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total if total > 0 else 0.0
        train_auc = roc_auc_score(train_targets, train_probs) if len(set(train_targets)) > 1 else float('nan')
        train_precision = precision_score(train_targets, train_preds, average="binary", zero_division=0)
        train_recall = recall_score(train_targets, train_preds, average="binary", zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, average="binary", zero_division=0)
        logging.info(
            f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Train precision = {train_precision:.4f}, "
            f"Train recall = {train_recall:.4f}, Train F1 = {train_f1:.4f}"
        )
        scheduler.step()

        # Gradient sanity check for first epoch
        if epoch == 0:
            any_backbone_grad = any(p.grad is not None for n, p in model.named_parameters() if "head" not in n)
            any_head_grad = any(p.grad is not None for n, p in model.named_parameters() if "head" in n)
            logging.info(f"[Grad check] backbone grads present? {any_backbone_grad}, head grads present? {any_head_grad}")
        
        # Validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_probs, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for inputs, targets, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(enabled=amp_enabled):
                    if args.model_architecture == 'endofm_lv':
                        # EndoFM-LV uses separate backbone and head
                        features = model(inputs)[0]      # backbone output
                        outputs = model.head(features)   # classification head
                    else:
                        # Alternative architectures
                        outputs = model(inputs)
                    loss = criterion(outputs, targets.long())
                running_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]   # the probability of class 1 (polyp); the probabilities of class 1 and 2 sum up to 1.0 
                preds = outputs.argmax(dim=1)                 # the predicted class (0=healthy or 1=polyp)
                correct += (preds.squeeze() == targets.int()).sum().item()
                total += targets.size(0)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.view(-1).detach().cpu().tolist())
                all_targets.extend(targets.view(-1).cpu().tolist())
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        val_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else float('nan')
        val_precision = precision_score(all_targets, all_preds, average="binary", zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average="binary", zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
        logging.info(
            f"Epoch {epoch+1}: Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}, Val AUC = {val_auc:.4f}, "
            f"Val precision = {val_precision:.4f}, Val recall = {val_recall:.4f}, Val F1 = {val_f1:.4f}"
        )

        # Log metrics
        record = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        }
        metrics_records.append(record)
        write_header = not os.path.exists(metrics_dir)
        with open(metrics_dir, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        # Monitor improvement
        if args.monitor == "val_loss":
            current = val_loss
        elif args.monitor == "val_acc":
            current = val_acc
        elif args.monitor == "val_auc":
            current = val_auc
        elif args.monitor == "val_precision":
            current = val_precision
        elif args.monitor == "val_recall":
            current = val_recall
        else:
            current = val_f1
        improved = (current < (best_metric - args.min_delta)) if args.monitor == "val_loss" else (current > (best_metric + args.min_delta))
        if improved:
            best_metric = current
            best_epoch = epoch+1
            epochs_since_improve = 0
            # Save the best model
            save_dict = save_finetuned_model(model, epoch+1, optimizer, scheduler, amp_enabled, scaler, args, best_metric)
            torch.save(save_dict, best_model_dir)
            logging.info(f"[Checkpoint] Saved best model to {best_model_dir} ({args.monitor}={best_metric:.4f})")
            # Generate the best model performance plots on validation dataset
            evaluate_model_performance(model, val_loader, device, run_dir, args.model_architecture, details="best_f1_") 
        else:
            epochs_since_improve += 1

        # Early stopping
        if args.patience > 0 and epochs_since_improve >= args.patience:
            logging.info(f"Early stopping at epoch {epoch+1} (no improvement in {args.patience} epochs).")
            stopped_early = True
            break

    return metrics_records, stopped_early, best_epoch, best_metric


def plot_training_metrics(metrics_records:list[dict], run_dir:str):
    """Generate comprehensive visualization plots of metric progression over training epochs."""
    epochs = [r["epoch"] for r in metrics_records]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["train_loss"] for r in metrics_records], 'o-', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, [r["val_loss"] for r in metrics_records], 's-', label='Val', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), bbox_inches='tight')
    plt.close()

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["train_acc"] for r in metrics_records], 'o-', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, [r["val_acc"] for r in metrics_records], 's-', label='Val', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curves.png"), bbox_inches='tight')
    plt.close()

    # Precision curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["train_precision"] for r in metrics_records], 'o-', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, [r["val_precision"] for r in metrics_records], 's-', label='Val', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "precision_curves.png"), bbox_inches='tight')
    plt.close()

    # Recall curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["train_recall"] for r in metrics_records], 'o-', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, [r["val_recall"] for r in metrics_records], 's-', label='Val', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "recall_curves.png"), bbox_inches='tight')
    plt.close()

    # F1 curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["train_f1"] for r in metrics_records], 'o-', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, [r["val_f1"] for r in metrics_records], 's-', label='Val', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('F1 score', fontsize=12)
    ax.set_title('F1 score Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "f1_curves.png"), bbox_inches='tight')
    plt.close()
    
    # AUC curve
    auc_values_train = [r["train_auc"] for r in metrics_records if not np.isnan(r["train_auc"])]
    auc_values_val = [r["val_auc"] for r in metrics_records if not np.isnan(r["val_auc"])]
    auc_values = [auc_values_train, auc_values_val]
    if auc_values:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs[:len(auc_values[0])], auc_values[0], 'o-', label='Train', linewidth=2, markersize=6)
        ax.plot(epochs[:len(auc_values[1])], auc_values[1], 's-', label='Val', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('AUC-ROC Over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "auc_curves.png"), bbox_inches='tight')
        plt.close()


def append_run_summary(summary_csv_path:str, record:dict) -> None:
    """Append one run-level summary row to the global experiment CSV."""
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    write_header = not os.path.exists(summary_csv_path)
    with open(summary_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def save_finetuning_data(
    args:argparse.Namespace, 
    metrics_records:list, 
    run_name:str, 
    frame_indices:list, 
    aug_config:dict, 
    duration:str, 
    stopped_early:bool, 
    epochs_completed:str,
    best_epoch:str,  
    device:torch.device, 
    best_epoch_val_loss:str,
    best_epoch_val_acc:str,
    best_epoch_val_auc:str,
    best_epoch_val_f1:str,
    best_epoch_val_precision:str,
    best_epoch_val_recall:str,
    save_dir:str,
):
    """Append one run-level row to the global experiment registry."""
    if metrics_records:
        run_record = {
            "run_name": run_name,
            "device": str(device),
            "model_architecture": args.model_architecture,
            "prep_endofm_lv_model": args.prep_endofm_lv_model,
            "unfreeze_endofm_lv_blocks": args.unfreeze_endofm_lv_blocks,
            "lr": args.lr,
            "clip_grad_norm": args.clip_grad_norm,
            "batch_size": args.batch_size,
            "use_saliency_loss": args.use_saliency_loss,
            "saliency_loss_alpha": args.alpha,
            "seed": args.seed,
            "cross_validation": args.cross_val,
            "n_folds": args.n_folds,
            "balance_train_data": args.balance_train_data,
            "num_frames": args.num_frames,
            "frame_indices": " ".join(str(i) for i in frame_indices),
            "temporal_reverse_prob": args.temporal_reverse_prob,
            "aug_config": aug_config,
            "duration_sec": duration,                               # For cross-validation, this is the average + std across folds
            "status": "completed",
            "stopped_early": stopped_early,                         # For cross validation, this is always set to True
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,                   # For cross-validation, this is the average + std across folds
            "best_epoch": best_epoch,                               # For cross-validation, this is the average + std across folds
            "monitor": args.monitor,
            "best_epoch_val_loss": best_epoch_val_loss,             # For cross-validation, this is the average + std across folds
            "best_epoch_val_acc": best_epoch_val_acc,               # For cross-validation, this is the average + std across folds
            "best_epoch_val_auc": best_epoch_val_auc,               # For cross-validation, this is the average + std across folds
            "best_epoch_val_f1": best_epoch_val_f1,                 # For cross-validation, this is the average + std across folds
            "best_epoch_val_precision": best_epoch_val_precision,   # For cross-validation, this is the average + std across folds
            "best_epoch_val_recall":best_epoch_val_recall,          # For cross-validation, this is the average + std across folds
        }
        append_run_summary(save_dir, run_record)
        logging.info(f"Appended run summary to {save_dir}")
    else:
        logging.warning("No epoch metrics were recorded; skipped experiment_runs.csv append.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning a video classification model on PolypGen6 data.')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'], help='Compute device selection.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # Cross-validation
    parser.add_argument('--cross_val', action='store_true', help='Whether to conduct cross validation instead of using the predefined train-val data division.')
    parser.add_argument('--n_folds', type=int, default=5, help='The number of folds for cross validation.')
    # Finetuning parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs of training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of subclips in each batch')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--use_saliency_loss', action='store_true', help='Whether to use saliency loss in addition to the classification loss.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for saliency loss')
    parser.add_argument("--lr", type=float, default=0.001, help='Start learning rate (for EndoFM-LV backbone it will be lower).')
    parser.add_argument('--monitor', type=str, default='val_f1', 
                        choices=['val_loss', 'val_acc', 'val_auc', 'val_precision', 'val_recall', 'val_f1'], 
                        help='Metric to monitor for early stopping/best checkpoint.')
    parser.add_argument('--patience', type=int, default=5, help='Epochs with no improvement before early stopping; set 0 to disable.')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum change to qualify as improvement.')
    parser.add_argument('--amp', action='store_true', help='Enable AMP (CUDA only).')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Grad norm clip; set 0 to disable.')
    # Model parameters
    parser.add_argument('--model_architecture', type=str, default='endofm_lv', 
                        choices=['endofm_lv', 'x3d'], help='The model architecture used for training on PolypGen6 video classification task.')
    parser.add_argument('--prep_endofm_lv_model', type=str, default='finetuned_backboneOnly',
                        choices=['finetuned_backboneOnly', 'finetuned', 'original_backbone'], 
                        help='How to initialize the model weights of EndoFM-LV.')
    parser.add_argument('--unfreeze_endofm_lv_blocks', type=int, default=0, 
                        help='Number of final transformer blocks of EndoFM-LV model to unfreeze; 0 to freeze full backbone.')
    # Data parameters
    parser.add_argument('--num_frames', type=int, default=6, help='Number of frames per clip.')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=None, help='Optional explicit frame indices to load, e.g. --frame_indices 1 4 6.')
    parser.add_argument('--balance_train_data', type=bool, default=False, help='Whether to use perfectly balanced train data.')
    parser.add_argument('--samples_per_class', type=int, default=60, help='The amount of input samples to select per class when train data balancing enabled.')
    # Temporal augmentation (clip-level)
    parser.add_argument('--temporal_reverse_prob', type=float, default=0.0, help='Probability to reverse the frame order during training.')
    # Flip augmentations (frame-level but identical for whole clip)
    parser.add_argument('--horizontal_flip_prob', type=float, default=0.5, help='Probability of horizontal flip.')
    parser.add_argument('--vertical_flip_prob', type=float, default=0.0, help='Probability of vertical flip.')
    # Color jitter augmentations (frame-level but identical for whole clip)
    parser.add_argument('--brightness_jitter', type=float, default=0.2, help='Brightness jitter strength (0=off).')
    parser.add_argument('--contrast_jitter', type=float, default=0.2, help='Contrast jitter strength (0=off).')
    parser.add_argument('--saturation_jitter', type=float, default=0.2, help='Saturation jitter strength (0=off).')
    parser.add_argument('--hue_jitter', type=float, default=0.0, help='Hue jitter strength (0=off).')
    # Blur augmentations (frame-level but identical for whole clip)
    parser.add_argument('--gaussian_blur_prob', type=float, default=0.2, help='Probability of Gaussian blur.')
    parser.add_argument('--motion_blur_prob', type=float, default=0.0, help='Probability of motion blur.')
    # Geometric augmentations (frame-level but identical for whole clip)
    parser.add_argument('--rotation_prob', type=float, default=0.0, help='Probability of rotation.')
    parser.add_argument('--rotation_degrees', type=float, default=15.0, help='Max rotation degrees (±degrees).')
    parser.add_argument('--random_crop_prob', type=float, default=0.0, help='Probability of random crop.')
    parser.add_argument('--perspective_prob', type=float, default=0.0, help='Probability of perspective transform.')
    # Noise augmentations (frame-level but identical for whole clip)
    parser.add_argument('--gaussian_noise_prob', type=float, default=0.2, help='Probability of Gaussian noise.')
    parser.add_argument('--gaussian_noise_std', type=float, default=0.05, help='Gaussian noise standard deviation.')
    
    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num_frames must be > 0.")
    if args.frame_indices is not None:
        if len(args.frame_indices) != args.num_frames:
            parser.error("--num_frames must match the number of entries in --frame_indices.")
        if any(idx < 1 for idx in args.frame_indices):
            parser.error("--frame_indices values must be >= 1.")

    # Enable logging messages
    logging.basicConfig(level=logging.INFO)

    # Set the compute device
    device = pick_device(args.device)
    pin_mem = device.type == "cuda"  # pin_memory helps only with CUDA
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=pin_mem, persistent_workers=args.num_workers > 0)
    logging.info(f"Using device {device}")

    # Set seeds for reproducibility
    random.seed(args.seed) 
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    BASE_SEED = args.seed
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Set AMP
    amp_enabled = args.amp and (device.type == "cuda")
    if args.amp and not amp_enabled:
        logging.info("AMP requested but CUDA not available; running without AMP.")
    scaler = GradScaler(enabled=amp_enabled)

    # Set main directories to save results
    now = datetime.now()
    run_name = f"{args.model_architecture}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    base_dir = "./model/model_results/finetuning"
    
    if args.cross_val:
        save_dir = os.path.join(base_dir, "cross_validation")
        global_runs_csv = os.path.join(save_dir, "average_experiment_runs.csv")
        run_dir = os.path.join(save_dir, run_name)
    else:
        save_dir = os.path.join(base_dir, "single_finetuning")
        global_runs_csv = os.path.join(save_dir, "experiment_runs.csv")
        run_dir = os.path.join(save_dir, run_name)
    
    if os.path.exists(run_dir):
        raise FileExistsError("The run_dir already exists.")
    os.makedirs(run_dir, exist_ok=True)

    run_start = datetime.now()

    # Build augmentation configuration
    aug_config = {
        'horizontal_flip_prob': args.horizontal_flip_prob,
        'vertical_flip_prob': args.vertical_flip_prob,
        'brightness_jitter': args.brightness_jitter,
        'contrast_jitter': args.contrast_jitter,
        'saturation_jitter': args.saturation_jitter,
        'hue_jitter': args.hue_jitter,
        'gaussian_blur_prob': args.gaussian_blur_prob,
        'motion_blur_prob': args.motion_blur_prob,
        'rotation_prob': args.rotation_prob,
        'rotation_degrees': args.rotation_degrees,
        'random_crop_prob': args.random_crop_prob,
        'perspective_prob': args.perspective_prob,
        'gaussian_noise_prob': args.gaussian_noise_prob,
        'gaussian_noise_std': args.gaussian_noise_std,
    }
    
    # Prepare the PolypGen6 training and valdation data
    logging.info("Preparing the datasets.")
    cfg = CfgPolypGen6()
    cfg.DATA.NUM_FRAMES = args.num_frames
    frame_indices = args.frame_indices if args.frame_indices is not None else list(range(1, args.num_frames + 1))

    datasets = get_polypgen_datasets(
        num_frames=cfg.DATA.NUM_FRAMES,
        frame_indices=frame_indices,
        load_masks=args.use_saliency_loss,
        temporal_reverse_prob=args.temporal_reverse_prob,
        aug_config=aug_config,
        apply_cross_validation=args.cross_val,
        n_splits=args.n_folds,
        random_state=args.seed,
    )

    # ---------------------------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # ---------------------------------------------------------------------------------------
    if args.cross_val:
        folds, _ = datasets   # folds is a list of train-val dataset divisions [(train_ds1, val_ds1), (train_ds2, val_ds2), ...]

        folds_duration = []
        folds_epochs_completed = []
        folds_best_epoch = []
        folds_best_epoch_val_loss = []
        folds_best_epoch_val_acc = []
        folds_best_epoch_val_auc = []
        folds_best_epoch_val_f1 = []
        folds_best_epoch_val_precision = []
        folds_best_epoch_val_recall = []

        for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
            logging.info(f"===== Starting Fold {fold_idx+1}/{args.n_folds} =====")

            # Set the directories for saving the results
            fold_dir = os.path.join(run_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)
            metrics_dir = os.path.join(fold_dir, "metrics.csv")
            best_model_dir = os.path.join(fold_dir, f"best_{args.monitor}.pth")

            # Balance the training data if requested
            if args.balance_train_data:
                logging.info("Class-balancing the training data.")
                train_dataset = get_balanced_subset(dataset=train_dataset, per_class=args.samples_per_class)
            
            # Create data loaders for this fold
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                worker_init_fn=seed_worker, collate_fn=collate_polypgen_batch, **loader_kwargs,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                worker_init_fn=seed_worker, collate_fn=collate_polypgen_batch, **loader_kwargs,
            )  # no augments on eval

            # Prepare for finetuning
            model, optimizer, scheduler, criterion = initialize_model(
                args=args, 
                cfg=cfg, 
                device=device, 
                train_loader=train_loader,
            )

            # Finetune the model
            metrics_records, stopped_early, best_epoch, best_metric = finetune_model(
                args=args, 
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                optimizer=optimizer, 
                amp_enabled=amp_enabled, 
                criterion=criterion, 
                scaler=scaler, 
                scheduler=scheduler, 
                device=device,
                run_dir=fold_dir,
                metrics_dir=metrics_dir,
                best_model_dir=best_model_dir,
            )

            run_end = datetime.now()
            duration = int((run_end - run_start).total_seconds())

            # Generate comprehensive training plots
            plot_training_metrics(metrics_records=metrics_records, run_dir=fold_dir)

            # Save the accuracy, F1 score, precision, and recall for val data using the best model
            folds_duration.append(duration)
            folds_epochs_completed.append(len(metrics_records))
            folds_best_epoch.append(best_epoch)
            folds_best_epoch_val_loss.append(metrics_records[best_epoch-1]["val_loss"])
            folds_best_epoch_val_acc.append(metrics_records[best_epoch-1]["val_acc"])
            folds_best_epoch_val_auc.append(metrics_records[best_epoch-1]["val_auc"])
            folds_best_epoch_val_f1.append(metrics_records[best_epoch-1]["val_f1"])
            folds_best_epoch_val_precision.append(metrics_records[best_epoch-1]["val_precision"])
            folds_best_epoch_val_recall.append(metrics_records[best_epoch-1]["val_recall"])
        
        # Save the (average) meta data of the cross-validation run
        save_finetuning_data(
            args=args, 
            metrics_records=metrics_records, 
            run_name=run_name, 
            frame_indices=frame_indices, 
            aug_config=aug_config, 
            duration=f"{np.mean(folds_duration)} ± {np.std(folds_duration)}", 
            stopped_early=True, 
            epochs_completed=f"{np.mean(folds_epochs_completed)} ± {np.std(folds_epochs_completed)}",
            best_epoch=f"{np.mean(folds_best_epoch)} ± {np.std(folds_best_epoch)}",  
            device=device, 
            best_epoch_val_loss=f"{np.mean(folds_best_epoch_val_loss)} ± {np.std(folds_best_epoch_val_loss)}",
            best_epoch_val_acc=f"{np.mean(folds_best_epoch_val_acc)} ± {np.std(folds_best_epoch_val_acc)}",
            best_epoch_val_auc=f"{np.mean(folds_best_epoch_val_auc)} ± {np.std(folds_best_epoch_val_auc)}",
            best_epoch_val_f1=f"{np.mean(folds_best_epoch_val_f1)} ± {np.std(folds_best_epoch_val_f1)}",
            best_epoch_val_precision=f"{np.mean(folds_best_epoch_val_precision)} ± {np.std(folds_best_epoch_val_precision)}",
            best_epoch_val_recall=f"{np.mean(folds_best_epoch_val_recall)} ± {np.std(folds_best_epoch_val_recall)}",
            save_dir=global_runs_csv,
        )
    
    # ---------------------------------------------------------------------------------------
    # SINGLE DATASET DIVISION 
    # Note: The val data includes more difficult examples compared to train and test datasets.
    # ---------------------------------------------------------------------------------------
    else:
        train_dataset, val_dataset, _ = datasets

        # Set the directories for saving the results
        metrics_dir = os.path.join(run_dir, "metrics.csv")
        best_model_dir = os.path.join(run_dir, f"best_{args.monitor}.pth")

        # Balance the training data if requested
        if args.balance_train_data:
            logging.info("Class-balancing the training data.")
            train_dataset = get_balanced_subset(dataset=train_dataset, per_class=args.samples_per_class)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            worker_init_fn=seed_worker, collate_fn=collate_polypgen_batch, **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            worker_init_fn=seed_worker, collate_fn=collate_polypgen_batch, **loader_kwargs,
        )  # no augments on eval

        # Prepare for finetuning
        model, optimizer, scheduler, criterion = initialize_model(
            args=args, 
            cfg=cfg, 
            device=device, 
            train_loader=train_loader,
        )

        # Finetune the model
        metrics_records, stopped_early, best_epoch, best_metric = finetune_model(
            args=args, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            amp_enabled=amp_enabled, 
            criterion=criterion, 
            scaler=scaler, 
            scheduler=scheduler, 
            device=device,
            run_dir=run_dir,
            metrics_dir=metrics_dir,
            best_model_dir=best_model_dir,
        )

        run_end = datetime.now()
        duration = int((run_end - run_start).total_seconds())

        # Generate comprehensive training plots
        plot_training_metrics(metrics_records=metrics_records, run_dir=run_dir)

        # Save the meta data of this finetuning run
        save_finetuning_data(
            args=args, 
            metrics_records=metrics_records, 
            run_name=run_name, 
            frame_indices=frame_indices, 
            aug_config=aug_config, 
            duration=str(duration), 
            stopped_early=stopped_early, 
            epochs_completed=str(len(metrics_records)),
            best_epoch=str(best_epoch),  
            device=device, 
            best_epoch_val_loss=str(metrics_records[best_epoch-1]["val_loss"]),
            best_epoch_val_acc=str(metrics_records[best_epoch-1]["val_acc"]),
            best_epoch_val_auc=str(metrics_records[best_epoch-1]["val_auc"]),
            best_epoch_val_f1=str(metrics_records[best_epoch-1]["val_f1"]),   
            best_epoch_val_precision=str(metrics_records[best_epoch-1]["val_precision"]),
            best_epoch_val_recall=str(metrics_records[best_epoch-1]["val_recall"]), 
            save_dir=global_runs_csv,
        )