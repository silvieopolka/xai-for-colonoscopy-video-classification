"""This script compares the pretrained and finetuned EndoFM-LV models on the test dataset."""

import argparse
import csv
import os
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.polypGen6.config import CfgPolypGen6
from data.process_polypgen_data import get_polypgen_datasets, collate_polypgen_batch
from model.eval_model_performance import evaluate_model_performance 
from model.helpers import pick_device
from model.model_loading import load_classification_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing / comparing EndoFM-LV models')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help='Compute device.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference.')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 recommended on macOS).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()

    # Enable logging messages
    logging.basicConfig(level=logging.INFO)

    # Set the compute device
    device = pick_device(args.device)

    # Set seeds for reproducibility
    random.seed(args.seed) 
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)

    # Prepare the PolypGen6 test data
    cfg = CfgPolypGen6()
    _, _, test_dataset = get_polypgen_datasets(num_frames=cfg.DATA.NUM_FRAMES, run_sanity_check=False)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_polypgen_batch,
                             num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Load pretrained model (PolypDiag-finetuned EndoFM-LV) and finetuned model (pretrained model, finetuned on PolypGen6)
    pretrained_model = load_classification_model(cfg=cfg, model="pretrained").to(device)
    finetuned_model = load_classification_model(cfg=cfg, model="finetuned").to(device)

    # Run inference and evaluate
    base_dir = "./model/model_results"
    save_dir = os.path.join(base_dir, "model_comparison")
    pretrained_save_dir = os.path.join(save_dir, "pretrained_endofm_lv")
    finetuned_save_dir = os.path.join(save_dir, "finetuned_endofm_lv")
    os.makedirs(pretrained_save_dir, exist_ok=True)
    os.makedirs(finetuned_save_dir, exist_ok=True)

    (ids, 
     labels, pretrained_polyp_probs, 
     acc_pretrained, f1_pretrained, 
     precision_pretrained, recall_pretrained) = evaluate_model_performance(pretrained_model, test_loader, device, save_dir=pretrained_save_dir, model_architecture='endofm_lv')

    (ids2, 
     labels2, finetuned_polyp_probs, 
     acc_finetuned, f1_finetuned, 
     precision_finetuned, recall_finetuned) = evaluate_model_performance(finetuned_model, test_loader, device, save_dir=finetuned_save_dir, model_architecture='endofm_lv')

    if not np.array_equal(labels, labels2):
        raise ValueError("Old and new scores do not share the same sequence of labels.")
    if not np.array_equal(ids, ids2):
        raise ValueError("Old and new scores do not share the same sequence of video ids.")

    print(f"[PRETRAINED ENDOFM-LV] Accuracy: {acc_pretrained}, F1 Score: {f1_pretrained}, Precision: {precision_pretrained}, Recall: {recall_pretrained}")
    print(f"[FINETUNED ENDOFM-LV]  Accuracy: {acc_finetuned}, F1 Score: {f1_finetuned}, Precision: {precision_finetuned}, Recall: {recall_finetuned}")

    csv_path = os.path.join(save_dir, "model_comparison_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "pretrained_polyp_probs", "finetuned_polyp_probs"])
        for vid, lbl, probs_p, probs_f in zip(ids, labels, pretrained_polyp_probs, finetuned_polyp_probs):
            writer.writerow([vid, lbl, probs_p, probs_f])