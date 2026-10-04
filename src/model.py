# -*- coding: utf-8 -*-
"""Off-Road Semantic Segmentation — Model, Loss, EMA & Checkpoint Loader.

Provides:
    - ``build_model``       – SegFormer MiT-B3 + U-Net decoder via SMP
    - ``HybridSegLoss``     – Lovász-Softmax + Cross-Entropy + Dice composite
    - ``EMA``               – Exponential Moving Average weight shadow
    - ``load_checkpoint``   – Safe checkpoint loader with user-friendly errors
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segmentation-models-pytorch is required: "
        "pip install segmentation-models-pytorch"
    ) from exc

from .dataset import IGNORE_INDEX, NUM_CLASSES

# ──────────────────────────────────────────────────────────────
#  Model Construction
# ──────────────────────────────────────────────────────────────

DEFAULT_ENCODER: str = "mit_b3"
DEFAULT_WEIGHTS: str = "imagenet"
DEFAULT_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints",
    "best_miou_model_v2.pth",
)


def build_model(
    encoder_name: str = DEFAULT_ENCODER,
    encoder_weights: Optional[str] = DEFAULT_WEIGHTS,
    num_classes: int = NUM_CLASSES,
    in_channels: int = 3,
) -> nn.Module:
    """Instantiate a SegFormer-encoder + U-Net decoder segmentation model.

    Parameters
    ----------
    encoder_name : str
        SMP encoder identifier (default: ``mit_b3`` / SegFormer-B3).
    encoder_weights : str or None
        Pretrained encoder weights (``"imagenet"`` or ``None``).
    num_classes : int
        Number of output semantic classes.
    in_channels : int
        Number of input image channels.

    Returns
    -------
    nn.Module
        Compiled segmentation model.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


# ──────────────────────────────────────────────────────────────
#  Composite Loss Function
# ──────────────────────────────────────────────────────────────

class HybridSegLoss(nn.Module):
    """Composite segmentation loss: Lovász-Softmax + Cross-Entropy + Dice.

    The Lovász-Softmax term directly optimises the Jaccard / mIoU index,
    while Cross-Entropy provides stable per-pixel gradients and the Dice
    term improves region-level overlap — especially for minority classes.

    Parameters
    ----------
    ce_weight : float
        Scalar weight for the Cross-Entropy term.
    lovasz_weight : float
        Scalar weight for the Lovász-Softmax term.
    dice_weight : float
        Scalar weight for the Dice term.
    ignore_index : int
        Label index to ignore (e.g. 255 for void/unlabelled).
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = smp.losses.LovaszLoss(
            mode="multiclass", from_logits=True, ignore_index=ignore_index
        )
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=ignore_index
        )
        self.w_ce = ce_weight
        self.w_lov = lovasz_weight
        self.w_dice = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted composite loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output of shape ``(B, C, H, W)``.
        targets : torch.Tensor
            Ground-truth class indices of shape ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        logits = logits.float()
        loss = (
            self.w_ce * self.ce(logits, targets)
            + self.w_lov * self.lovasz(logits, targets)
            + self.w_dice * self.dice(logits, targets)
        )
        return loss


# ──────────────────────────────────────────────────────────────
#  Exponential Moving Average (EMA)
# ──────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average shadow of a model's weights.

    Maintains a running exponential average of model parameters, which
    typically generalises better than the raw optimised weights.

    Parameters
    ----------
    model : nn.Module
        Source model whose parameters are tracked.
    decay : float
        EMA decay rate (default: 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module: nn.Module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model weights.

        Parameters
        ----------
        model : nn.Module
            The actively training model.
        """
        model_state = model.state_dict()
        for key, shadow_param in self.module.state_dict().items():
            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(self.decay).add_(
                    model_state[key].detach(), alpha=1.0 - self.decay
                )
            else:
                shadow_param.copy_(model_state[key])


# ──────────────────────────────────────────────────────────────
#  Checkpoint Loader
# ──────────────────────────────────────────────────────────────

def load_checkpoint(
    path: Optional[str] = None,
    device: Optional[torch.device] = None,
    encoder_name: str = DEFAULT_ENCODER,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Load a trained checkpoint into a fresh model instance.

    Parameters
    ----------
    path : str, optional
        Path to ``*.pth`` state-dict file.  Defaults to
        ``checkpoints/best_miou_model_v2.pth`` relative to the repo root.
    device : torch.device, optional
        Target device for the model.  Defaults to CUDA if available.
    encoder_name : str
        Encoder name matching the saved checkpoint architecture.
    num_classes : int
        Number of output classes matching the saved checkpoint.

    Returns
    -------
    nn.Module
        Model loaded with pretrained weights, set to ``eval()`` mode.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist, with download instructions.
    """
    if path is None:
        path = DEFAULT_CHECKPOINT_PATH

    if not os.path.isfile(path):
        print(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  ERROR: Model checkpoint not found!                     ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            f"║  Expected: {path:<45s}║\n"
            "║                                                          ║\n"
            "║  Download the weights and place them at:                 ║\n"
            "║    checkpoints/best_miou_model_v2.pth                   ║\n"
            "║                                                          ║\n"
            "║  Automated download:                                     ║\n"
            "║    pip install gdown                                     ║\n"
            "║    gdown --id YOUR_FILE_ID \\                             ║\n"
            "║      -O checkpoints/best_miou_model_v2.pth              ║\n"
            "║                                                          ║\n"
            "║  See checkpoints/README.md for full instructions.       ║\n"
            "╚══════════════════════════════════════════════════════════╝\n",
            file=sys.stderr,
        )
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "See checkpoints/README.md for download instructions."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        encoder_name=encoder_name,
        encoder_weights=None,  # Loading from checkpoint, not ImageNet
        num_classes=num_classes,
    )
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"✓ Loaded checkpoint: {path}  →  device={device}")
    return model
