# -*- coding: utf-8 -*-
"""Off-Road Semantic Segmentation — Multi-Scale TTA Inference & Mask Export.

CLI Usage
---------
    python -m src.inference \
        --input-dir  data/test/Color_Images \
        --output-dir predicted_masks \
        --checkpoint checkpoints/best_miou_model_v2.pth

Applies multi-scale Test-Time Augmentation (TTA) at 0.75×, 1.0×, 1.25× with
horizontal flipping, then exports single-channel 8-bit PNG masks at the
native 960×540 resolution using nearest-neighbour upscaling.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import H, W, MEAN, NUM_CLASSES, STD
from .model import load_checkpoint


# ──────────────────────────────────────────────────────────────
#  Multi-Scale TTA Engine
# ──────────────────────────────────────────────────────────────

_MEAN_NP: np.ndarray = np.array(MEAN, dtype=np.float32)
_STD_NP: np.ndarray = np.array(STD, dtype=np.float32)

# Default TTA scales
DEFAULT_SCALES: Tuple[float, ...] = (0.75, 1.0, 1.25)

# Native output resolution (original camera frame)
NATIVE_H: int = 540
NATIVE_W: int = 960


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    scales: Sequence[float] = DEFAULT_SCALES,
    flip: bool = True,
) -> np.ndarray:
    """Run multi-scale TTA inference on a single RGB image.

    Parameters
    ----------
    model : nn.Module
        Segmentation model in eval mode.
    image_rgb : np.ndarray
        Input image as ``(H, W, 3)`` uint8 RGB array.
    device : torch.device
        Computation device.
    scales : sequence of float
        Scale factors relative to the training resolution (544×960).
    flip : bool
        Whether to include horizontal-flip augmentation.

    Returns
    -------
    np.ndarray
        Predicted class-index mask of shape ``(H_orig, W_orig)`` as uint8.
    """
    model.eval()
    h_orig, w_orig = image_rgb.shape[:2]
    prob_accum = torch.zeros(
        1, NUM_CLASSES, h_orig, w_orig, device=device
    )

    for scale in scales:
        # Scale relative to training resolution, snap to multiples of 32
        h_s = int(round(H * scale / 32)) * 32
        w_s = int(round(W * scale / 32)) * 32

        resized = cv2.resize(
            image_rgb, (w_s, h_s), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0

        # Normalise and convert to tensor
        tensor = torch.from_numpy(
            ((resized - _MEAN_NP) / _STD_NP).transpose(2, 0, 1)
        ).unsqueeze(0).to(device)

        augmentations = [False]
        if flip:
            augmentations.append(True)

        for do_flip in augmentations:
            x = torch.flip(tensor, dims=[3]) if do_flip else tensor

            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logits = logits.float()

            if do_flip:
                logits = torch.flip(logits, dims=[3])

            # Bilinear upscale to original resolution for soft averaging
            logits = F.interpolate(
                logits,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False,
            )
            prob_accum += logits.softmax(dim=1)

    # Argmax over accumulated soft probabilities
    pred = prob_accum.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


# ──────────────────────────────────────────────────────────────
#  Batch Inference
# ──────────────────────────────────────────────────────────────

def run_batch_inference(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    scales: Sequence[float] = DEFAULT_SCALES,
    flip: bool = True,
    native_size: Optional[Tuple[int, int]] = (NATIVE_W, NATIVE_H),
) -> List[str]:
    """Run TTA inference on all images in a directory.

    Parameters
    ----------
    model : nn.Module
        Loaded segmentation model.
    input_dir : str
        Directory containing input RGB images.
    output_dir : str
        Directory for exported single-channel 8-bit PNG masks.
    device : torch.device
        Computation device.
    scales : sequence of float
        TTA scale factors.
    flip : bool
        Enable horizontal-flip TTA.
    native_size : tuple of (int, int) or None
        ``(width, height)`` to resize output masks via nearest-neighbour.
        ``None`` keeps the original image dimensions.

    Returns
    -------
    list of str
        Paths to all exported mask files.
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = sorted(
        f
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    )
    if not files:
        print(f"⚠  No image files found in {input_dir}", file=sys.stderr)
        return []

    print(f"\n  Input  : {input_dir}  ({len(files)} images)")
    print(f"  Output : {output_dir}")
    print(f"  Scales : {scales}  |  Flip: {flip}")
    if native_size:
        print(f"  Resize : → {native_size[0]}×{native_size[1]} (nearest)\n")
    else:
        print(f"  Resize : keep original\n")

    exported: List[str] = []

    for filename in tqdm(files, desc="Inference"):
        img_bgr = cv2.imread(os.path.join(input_dir, filename))
        if img_bgr is None:
            print(f"  ⚠ Skipping unreadable: {filename}", file=sys.stderr)
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = predict_tta(model, img_rgb, device, scales=scales, flip=flip)

        # Nearest-neighbour resize to native output resolution
        if native_size is not None:
            mask = cv2.resize(
                mask,
                native_size,
                interpolation=cv2.INTER_NEAREST,
            )

        out_path = os.path.join(
            output_dir, os.path.splitext(filename)[0] + ".png"
        )
        cv2.imwrite(out_path, mask)
        exported.append(out_path)

    print(f"\n✓ Exported {len(exported)} masks to {output_dir}\n")
    return exported


# ──────────────────────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-scale TTA inference for off-road semantic segmentation. "
            "Exports single-channel 8-bit PNG masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory of input RGB images.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="predicted_masks",
        help="Directory for exported prediction masks.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/best_miou_model_v2.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.75, 1.0, 1.25],
        help="TTA scale factors.",
    )
    parser.add_argument(
        "--no-flip", action="store_true",
        help="Disable horizontal-flip TTA.",
    )
    parser.add_argument(
        "--native-width", type=int, default=NATIVE_W,
        help="Output mask width (nearest-neighbour resize).",
    )
    parser.add_argument(
        "--native-height", type=int, default=NATIVE_H,
        help="Output mask height (nearest-neighbour resize).",
    )
    parser.add_argument(
        "--keep-size", action="store_true",
        help="Skip resize; keep original image dimensions.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for batch inference."""
    args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_checkpoint(path=args.checkpoint, device=device)

    native_size = (
        None if args.keep_size
        else (args.native_width, args.native_height)
    )

    exported = run_batch_inference(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=device,
        scales=tuple(args.scales),
        flip=not args.no_flip,
        native_size=native_size,
    )

    sys.exit(0 if exported else 1)


if __name__ == "__main__":
    main()
