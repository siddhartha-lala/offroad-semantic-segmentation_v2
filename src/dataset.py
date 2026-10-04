# -*- coding: utf-8 -*-
"""Off-Road Semantic Segmentation — Dataset & Augmentation Pipeline.

Provides:
    - Falcon simulation ID → contiguous class remapping
    - Albumentations train/val transforms at 544×960
    - ``OffroadDataset`` PyTorch Dataset
    - ``rare_image_weights`` minority-class sampler weighting
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "albumentations is required: pip install albumentations"
    ) from exc

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

NUM_CLASSES: int = 11
IGNORE_INDEX: int = 255

# Native training resolution: 544×960 (divisible by 32, ≈16:9)
H: int = 544
W: int = 960

# ImageNet normalisation
MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

# Raw Falcon simulation ID → contiguous class index
ID_MAP: dict[int, int] = {
    0: 0,       # Background / Void
    100: 1,     # Trees
    200: 2,     # Lush Bushes
    300: 3,     # Dry Grass
    500: 4,     # Dry Bushes
    550: 5,     # Ground Clutter
    600: 6,     # Flowers
    700: 7,     # Logs
    800: 8,     # Rocks
    7100: 9,    # Landscape / Dirt Trail
    10000: 10,  # Sky
}

CLASS_NAMES: Tuple[str, ...] = (
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape/Dirt", "Sky",
)

# ──────────────────────────────────────────────────────────────
#  Look-Up Table for vectorised mask remapping
# ──────────────────────────────────────────────────────────────

_LUT: np.ndarray = np.full(65536, fill_value=IGNORE_INDEX, dtype=np.uint8)
for _raw, _idx in ID_MAP.items():
    _LUT[_raw] = _idx


def load_mask(path: str) -> np.ndarray:
    """Load a single-channel segmentation mask and remap to class indices.

    Parameters
    ----------
    path : str
        Absolute or relative path to the mask image (16-bit or 8-bit PNG).

    Returns
    -------
    np.ndarray
        H×W uint8 array with values in ``[0, NUM_CLASSES-1]``.
    """
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return _LUT[mask]


# ──────────────────────────────────────────────────────────────
#  Albumentations Pipelines
# ──────────────────────────────────────────────────────────────

def get_train_transform() -> A.Compose:
    """Training augmentation pipeline with geometric & photometric jitter."""
    return A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=(-0.15, 0.3),
            rotate_limit=10,
            p=0.6,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4
        ),
        A.MotionBlur(blur_limit=3, p=0.15),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """Validation transform: resize + normalise only."""
    return A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class OffroadDataset(Dataset):
    """Off-road semantic segmentation dataset.

    Expects matching sorted file lists in ``img_dir`` and ``mask_dir``
    (stem names must agree).

    Parameters
    ----------
    img_dir : str
        Directory of RGB images (.png / .jpg).
    mask_dir : str
        Directory of raw Falcon segmentation masks.
    transform : albumentations.Compose, optional
        Augmentation pipeline applied to both image and mask.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.img_files: List[str] = sorted(os.listdir(img_dir))
        self.mask_files: List[str] = sorted(os.listdir(mask_dir))

        if len(self.img_files) != len(self.mask_files):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self.img_files)} images vs {len(self.mask_files)} masks"
            )
        for img_name, mask_name in zip(self.img_files, self.mask_files):
            if os.path.splitext(img_name)[0] != os.path.splitext(mask_name)[0]:
                raise ValueError(
                    f"Filename stem mismatch: {img_name} ↔ {mask_name}"
                )

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.img_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        image = cv2.cvtColor(
            cv2.imread(img_path), cv2.COLOR_BGR2RGB
        )
        mask = load_mask(mask_path)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


# ──────────────────────────────────────────────────────────────
#  Minority-Class Sample Weighting
# ──────────────────────────────────────────────────────────────

def rare_image_weights(
    mask_dir: str,
    files: Sequence[str],
    rare_classes: Tuple[int, ...] = (4, 5, 6, 7, 8),
    boost: float = 1.5,
) -> List[float]:
    """Compute per-image sampling weights that up-weight rare hazard classes.

    Images containing more rare-class pixels receive higher weight so that
    a ``WeightedRandomSampler`` can balance the training distribution.

    Parameters
    ----------
    mask_dir : str
        Directory containing segmentation mask files.
    files : Sequence[str]
        Ordered list of mask filenames (matching dataset order).
    rare_classes : tuple of int
        Class indices considered rare (default: Dry Bushes, Ground Clutter,
        Flowers, Logs, Rocks).
    boost : float
        Additive weight per rare class present in the mask.

    Returns
    -------
    list of float
        Sampling weight for each image.
    """
    rare_set = set(rare_classes)
    weights: List[float] = []
    for filename in files:
        present = set(
            np.unique(load_mask(os.path.join(mask_dir, filename))).tolist()
        )
        weights.append(1.0 + boost * len(present & rare_set))
    return weights
