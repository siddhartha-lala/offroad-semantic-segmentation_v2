# Off-Road Semantic Segmentation — Assets
# ─────────────────────────────────────────

This directory stores visual assets referenced in the project README.

## Required File

| File | Description |
|------|-------------|
| `demo_overlay.png` | 3-panel showcase image: **Raw Input → Semantic Mask → 60/40 Alpha Blend** |

## How to Generate

The overlay is produced by **Cell 10** of the training notebook (`notebooks/offseg_improved.ipynb`).
Save the resulting matplotlib figure as `demo_overlay.png` and place it here:

```
assets/
└── demo_overlay.png
```

> **Note:** This file is committed to the repository and should be ≤ 5 MB for optimal GitHub rendering.
