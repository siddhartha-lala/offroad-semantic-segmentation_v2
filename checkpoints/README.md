# Pretrained Model Weights
# ─────────────────────────

This directory is the expected location for the trained model checkpoint.

## Download

| File | Size | mIoU | Download |
|------|------|------|----------|
| `best_miou_model_v2.pth` | ~180 MB | **71.70%** | [Google Drive ↗](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing) |

### Automated Download (via `gdown`)

```bash
pip install gdown
gdown --id YOUR_FILE_ID -O checkpoints/best_miou_model_v2.pth
```

### Manual Download

1. Download from the Google Drive link above.
2. Place the file in this directory:

```
checkpoints/
├── .gitkeep
├── README.md            ← you are here
└── best_miou_model_v2.pth   ← place here
```

## Why Not in Git?

The checkpoint is **~180 MB**, which exceeds GitHub's **100 MB hard per-file limit**.
It is explicitly excluded via `.gitignore`. Do **not** attempt to `git add` this file.

## Checkpoint Format

- **Framework:** PyTorch `state_dict` (not a full training checkpoint)
- **Architecture:** SegFormer MiT-B3 encoder + U-Net decoder (11 classes)
- **Origin:** EMA shadow weights after 40-epoch Lovász + CE + Dice training
- **Load Example:**

```python
from src.model import load_checkpoint
model = load_checkpoint("checkpoints/best_miou_model_v2.pth")
```
