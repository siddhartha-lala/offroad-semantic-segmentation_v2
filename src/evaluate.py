# -*- coding: utf-8 -*-
"""Off-Road Semantic Segmentation — Validation & Evaluation Script.

CLI Usage
---------
    python -m src.evaluate \
        --img-dir  data/val/Color_Images \
        --mask-dir data/val/Segmentation \
        --checkpoint checkpoints/best_miou_model_v2.pth

Outputs a per-class IoU breakdown, confusion matrix summary, and mean IoU.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    OffroadDataset,
    get_val_transform,
)
from .model import load_checkpoint


# ──────────────────────────────────────────────────────────────
#  Core Evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class IoU and confusion matrix over a DataLoader.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (already in eval mode).
    loader : DataLoader
        Validation data loader.
    device : torch.device
        Computation device.
    num_classes : int
        Total number of semantic classes.

    Returns
    -------
    mean_iou : float
        Mean IoU across classes present in the ground truth.
    iou : np.ndarray
        Per-class IoU array of shape ``(num_classes,)``.
    has_gt : np.ndarray
        Boolean mask indicating which classes appear in the GT.
    confusion : np.ndarray
        ``(num_classes, num_classes)`` confusion matrix.
    """
    model.eval()
    cm = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=device
    )

    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)

        logits = logits.float()
        preds = logits.argmax(dim=1)

        valid = masks != IGNORE_INDEX
        cm += torch.bincount(
            masks[valid] * num_classes + preds[valid],
            minlength=num_classes ** 2,
        ).view(num_classes, num_classes)

    # Per-class IoU
    intersection = cm.diag().float()
    union = cm.sum(dim=0).float() + cm.sum(dim=1).float() - intersection
    iou = (intersection / union.clamp(min=1)).cpu().numpy()

    has_gt = (cm.sum(dim=1) > 0).cpu().numpy()
    mean_iou = float(iou[has_gt].mean()) if has_gt.any() else 0.0

    return mean_iou, iou, has_gt, cm.cpu().numpy()


def print_report(
    mean_iou: float,
    iou: np.ndarray,
    has_gt: np.ndarray,
    confusion: np.ndarray,
) -> None:
    """Pretty-print evaluation results to stdout.

    Parameters
    ----------
    mean_iou : float
        Mean IoU value.
    iou : np.ndarray
        Per-class IoU array.
    has_gt : np.ndarray
        Boolean mask for classes with ground-truth pixels.
    confusion : np.ndarray
        Confusion matrix.
    """
    separator = "═" * 44

    print(f"\n╔{separator}╗")
    print(f"║  {'Per-Class IoU Breakdown':^40s}  ║")
    print(f"╠{separator}╣")

    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_id]
        if has_gt[cls_id]:
            score = f"{iou[cls_id] * 100:6.2f}%"
        else:
            score = "  n/a (not in GT)"
        print(f"║  {name:<22s} {score:>18s}  ║")

    print(f"╠{separator}╣")
    print(f"║  {'Overall mIoU':<22s} {mean_iou * 100:17.2f}%  ║")
    print(f"╚{separator}╝")

    # Top confusions
    print("\n  Top Confusions (true → predicted, % of true-class pixels):")
    print("  " + "─" * 50)
    rows = []
    for true_cls in range(NUM_CLASSES):
        row_sum = confusion[true_cls].sum()
        if row_sum == 0:
            continue
        for pred_cls in range(NUM_CLASSES):
            if true_cls != pred_cls:
                pct = confusion[true_cls, pred_cls] / row_sum * 100
                rows.append((pct, CLASS_NAMES[true_cls], CLASS_NAMES[pred_cls]))

    for pct, true_name, pred_name in sorted(rows, reverse=True)[:12]:
        print(f"    {true_name:<18s} → {pred_name:<18s} {pct:5.1f}%")
    print()


# ──────────────────────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate off-road segmentation model on a validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-dir", type=str, required=True,
        help="Path to validation RGB images directory.",
    )
    parser.add_argument(
        "--mask-dir", type=str, required=True,
        help="Path to validation segmentation masks directory.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/best_miou_model_v2.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model
    model = load_checkpoint(
        path=args.checkpoint, device=device
    )

    # Build validation set
    val_dataset = OffroadDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        transform=get_val_transform(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\nDataset : {len(val_dataset)} samples")
    print(f"Device  : {device}")
    print(f"Model   : {args.checkpoint}\n")

    # Evaluate
    mean_iou, iou, has_gt, confusion = evaluate(
        model, val_loader, device
    )
    print_report(mean_iou, iou, has_gt, confusion)

    # Exit code: non-zero if mIoU below 50% (CI gate)
    sys.exit(0 if mean_iou >= 0.50 else 1)


if __name__ == "__main__":
    main()
