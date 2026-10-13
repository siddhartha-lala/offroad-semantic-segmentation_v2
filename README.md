<div align="center">

# 🏜️ Off-Road Semantic Segmentation

### Autonomous Scene Understanding for Unstructured Desert Terrain

[![Architecture SegFormer MiT-B3 + U-Net](https://img.shields.io/badge/Model-SegFormer%20MiT--B3%20%2B%20U--Net-8A2BE2?style=for-the-badge)](https://github.com/qubvel-org/segmentation_models.pytorch)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SMP 0.3+](https://img.shields.io/badge/SMP-0.3%2B-00C7B7?style=for-the-badge)](https://github.com/qubvel-org/segmentation_models.pytorch)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Colab T4 Verified](https://img.shields.io/badge/Colab-T4%20Verified-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

*High-resolution semantic segmentation pipeline powered by a **SegFormer MiT-B3 + U-Net** architecture, achieving **71.70% mIoU** on 11 navigational terrain classes — a **+30.5 point gain** over standard cross-entropy baselines.*

</div>

> ⚡ **Model at a Glance:** **SegFormer MiT-B3** (Mix Transformer encoder) coupled with a **U-Net** decoder (multi-scale skip connections), optimized with **Lovász-Softmax + CE + Dice** composite loss at native **544×960** resolution.

---

## 🎯 Visual Results

<div align="center">

<img src="assets/demo_overlay.png" alt="3-Panel Demo: Raw Input | Semantic Mask | Alpha Blend Overlay" width="100%"/>

<sub><b>Left:</b> Raw camera input (960×540) &nbsp;|&nbsp; <b>Center:</b> Predicted semantic mask &nbsp;|&nbsp; <b>Right:</b> 60/40 alpha blend overlay</sub>

</div>

---

## 📊 Performance Comparison

| Metric | Standard CE Baseline | **Our Pipeline** | Δ Gain |
|:---|:---:|:---:|:---:|
| **Validation mIoU** | ~41.2% | **71.70%** | **+30.5** |
| Sky | ~88% | **98.92%** | +10.9 |
| Landscape / Dirt Trail | ~65% | **73.85%** | +8.9 |
| Trees | ~72% | **90.16%** | +18.2 |
| Lush Bushes | ~38% | **74.88%** | +36.9 |
| Dry Grass | ~35% | **72.95%** | +37.9 |
| Dry Bushes | ~18% | **54.38%** | +36.4 |
| Ground Clutter | ~12% | **48.97%** | +37.0 |
| Flowers | ~15% | **73.33%** | +58.3 |
| Logs | ~8% | **69.84%** | +61.8 |
| Rocks | ~10% | **59.71%** | +49.7 |

> **Key result:** Lovász-Softmax + Dice + Focal/CE composite loss with EMA weight tracking and rare-class sample weighting enables reliable obstacle isolation even for sparse hazard categories.

### 🔍 Error Analysis — Top Confusions

Dominant misclassification patterns on the validation set (% of true-class pixels predicted as another class):

| True Class | → Predicted As | Confusion % | Analysis |
|:---|:---|:---:|:---|
| Ground Clutter | → Landscape/Dirt | 25.2% | Textural similarity on sandy substrate |
| Dry Bushes | → Dry Grass | 12.9% | Desiccated vegetation overlap |
| Rocks | → Landscape/Dirt | 12.6% | Partially buried / color-matched rocks |
| Rocks | → Dry Grass | 10.3% | Small rock clusters in grassy terrain |
| Landscape/Dirt | → Dry Grass | 9.9% | Transitional terrain boundary |
| Ground Clutter | → Dry Grass | 9.3% | Low-lying debris vs. grass texture |
| Logs | → Landscape/Dirt | 7.7% | Fallen wood on dirt surface |
| Dry Grass | → Landscape/Dirt | 7.4% | Sparse grass / exposed soil |
| Dry Bushes | → Landscape/Dirt | 5.9% | Low scrub on trail edges |
| Flowers | → Landscape/Dirt | 5.3% | Isolated blooms in open terrain |
| Lush Bushes | → Dry Grass | 5.1% | Seasonal vegetation transition |
| Logs | → Dry Grass | 4.7% | Decomposing wood / grass boundary |

> **Insight:** The hardest confusions occur between semantically adjacent terrain classes (e.g., Ground Clutter ↔ Landscape/Dirt). From a navigation safety perspective, the critical chassis-hazard classes (Rocks, Logs) maintain **59.7%** and **69.8%** IoU respectively — sufficient for reliable avoidance planning.

---

## 🗺️ 11-Class Taxonomy & Navigational Impact Matrix

| ID | Class | Color | Nav Impact | Traversal Policy |
|:---:|:---|:---:|:---|:---|
| 0 | **Background / Void** | ⬛ Black | — | Ignore (annotation artifact) |
| 1 | **Trees** | 🟤 Dark Green | 🔴 Hard obstacle | Full avoidance — canopy & trunk collision |
| 2 | **Lush Bushes** | 🟢 Forest Green | 🟠 Soft obstacle | Slow traversal; potential entanglement |
| 3 | **Dry Grass** | 🫒 Olive | 🟡 Mild terrain | Passable — minor traction variation |
| 4 | **Dry Bushes** | 🟫 Saddle Brown | 🟠 Soft obstacle | Slow traversal; branch snag risk |
| 5 | **Ground Clutter** | 🫒 Olive Drab | 🟡 Mild penalty | Passable loose debris |
| 6 | **Flowers** | 🌸 Pink | 🟡 Mild penalty | Passable — conservation-aware slowdown |
| 7 | **Logs** | 🟫 Sienna | 🔴 Chassis hazard | **Hard avoidance** — undercarriage strike |
| 8 | **Rocks** | ⬜ Gray | 🔴 Chassis hazard | **Hard avoidance** — wheel/axle damage |
| 9 | **Landscape / Dirt** | 🟠 Sandy Brown | 🟢 Drivable | **Primary navigation corridor** |
| 10 | **Sky** | 🔵 Sky Blue | ⚪ Boundary | Horizon reference — attitude estimation |

---

## 🏗️ Architecture

```
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                    OFF-ROAD SEGMENTATION PIPELINE                            │
 └───────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
  │  High-Res Input  │    │  Encoder Backbone     │    │  U-Net Decoder           │
  │   544 × 960 RGB  │───▶│  SegFormer MiT-B3     │───▶│  Feature Fusion &        │
  │  (÷32 aligned)   │    │  (ImageNet pretrained) │    │  Skip Connections        │
  └─────────────────┘    └──────────────────────┘    └────────┬─────────────────┘
                                                              │
                          ┌───────────────────────────────────┘
                          ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                         COMPOSITE LOSS                                      │
  │  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐            │
  │  │ Lovász-Softmax│ + │ Cross-Entropy    │ + │ 0.5 × Dice Loss  │            │
  │  │ (mIoU proxy)  │   │ (pixel-wise CE)  │   │ (region overlap)  │            │
  │  └──────────────┘   └──────────────────┘   └──────────────────┘            │
  └────────────────────────────────┬─────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  TRAINING DYNAMICS                                                          │
  │  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐ │
  │  │ AdamW Optimizer │  │ Cosine Annealing  │  │  EMA Shadow (decay=0.999)   │ │
  │  │ (diff. LR:      │  │ + Linear Warmup   │  │  ema.update(model) each     │ │
  │  │  enc=6e-5       │  │ LambdaLR schedule  │  │  optimizer step             │ │
  │  │  dec=6e-4)      │  │                    │  │                             │ │
  │  └────────────────┘  └──────────────────┘  └──────────────────────────────┘ │
  │  ┌──────────────────────────────┐  ┌─────────────────────────────────┐      │
  │  │ AMP FP16 Mixed Precision     │  │ WeightedRandomSampler            │      │
  │  │ (GradScaler + autocast)      │  │ (rare class boost: 4,5,6,7,8)   │      │
  │  └──────────────────────────────┘  └─────────────────────────────────┘      │
  └──────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                        INFERENCE (TTA)                                      │
  │                                                                             │
  │  Multi-Scale:  0.75× ──┐                                                   │
  │                1.00× ──┼── Softmax Accumulate ──▶ Argmax ──▶ NN Resize     │
  │                1.25× ──┘      + H-Flip                       960 × 540     │
  │                                                                             │
  │  Output: Single-channel 8-bit PNG class-index mask                          │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
offroad-semantic-segmentation_v2/
├── .gitignore                  # Excludes *.pth, datasets, caches
├── requirements.txt            # Pinned dependencies
├── README.md                   # ← You are here
├── assets/
│   ├── README.md               # Asset placement instructions
│   └── demo_overlay.png        # 3-panel visual showcase
├── checkpoints/
│   ├── .gitkeep                # Preserves directory in Git
│   └── README.md               # Local checkpoint setup guide
├── notebooks/
│   └── README.md               # Colab notebook placement guide
└── src/
    ├── __init__.py
    ├── dataset.py              # OffroadDataset, augmentations, sampler weights
    ├── model.py                # Model builder, HybridSegLoss, EMA, checkpoint loader
    ├── evaluate.py             # CLI: confusion matrix & per-class IoU evaluation
    └── inference.py            # CLI: multi-scale TTA batch mask inference
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Built for autonomous ground vehicles navigating the unknown.*

</div>
