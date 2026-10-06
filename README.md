# DepthPro — Monocular Depth Estimation

> DPT + MiDaS v3.1 · δ1 93.2% · RMSE 0.412m on NYUv2 · SILog 10.8 on KITTI

## Overview

DepthPro estimates dense, metric-scale depth maps from a single RGB image — no LiDAR, no stereo rig. Fine-tuned across indoor and outdoor domains using a scale-invariant loss with gradient matching, it generalises from bedroom-scale to road-scale scenes without additional calibration.

---

## Results

| Benchmark | Metric | Score | Baseline (MiDaS v3.1) |
|---|---|---|---|
| NYU Depth v2 (indoor) | δ1 ↑ | **93.2%** | 90.1% |
| NYU Depth v2 | RMSE ↓ | **0.412m** | 0.479m |
| KITTI (outdoor) | SILog ↓ | **10.8** | 12.4 |
| KITTI | δ1 ↑ | **88.6%** | 85.2% |
| Inference latency | ms ↓ | **18ms** | 22ms |

---

## Architecture

```
RGB Image (H×W×3)
      │
  ViT-L/16 Encoder            ← ImageNet-21K pretrained
  (global attention at
   all resolutions)
      │
  Reassemble + Fusion         ← DPT multi-scale feature aggregation
  (4 stages: 1/4, 1/8,
   1/16, 1/32 resolution)
      │
  Dense Prediction Head       ← pixel-wise depth regression
      │
  Depth Map (H×W×1)           ← metric scale (metres)
```

### Key Design Decisions

**Scale-invariant loss with gradient matching** — Standard L1/L2 losses collapse to mean-depth predictions on high-dynamic-range outdoor scenes. The scale-invariant term (SILog) normalises per-image depth scale while the gradient matching term sharpens object boundaries, recovering fine edge detail that SILog alone averages away.

**Domain-randomised fine-tuning** — Joint training on NYUv2, KITTI, MatterPort3D, and DIODE with random crop scales prevents indoor–outdoor catastrophic forgetting. Separate normalisation statistics per dataset head; shared encoder weights.

**DPT vs. encoder-decoder CNNs** — ViT's global self-attention captures long-range spatial context at full resolution, eliminating the information bottleneck in U-Net-style skip connections. +3.1% δ1 over BTS (CVPR 2021) on NYUv2 with 30% fewer parameters.

---

## Dataset

| Dataset | Scenes | Depth range | Use |
|---|---|---|---|
| NYU Depth v2 | 464 indoor | 0–10m | Primary train/eval |
| KITTI | 61 outdoor drives | 0–80m | Outdoor eval |
| MatterPort3D | 90 indoor buildings | 0–10m | Fine-tuning |
| DIODE | Indoor + outdoor | 0–300m | Robustness eval |

---

## Tech Stack

`PyTorch` `DPT (Dense Prediction Transformer)` `MiDaS v3.1` `OpenCV` `ONNX` `NumPy`

---

## Use Cases

- **AR scene understanding** — real-time occlusion and surface reconstruction
- **Robotics navigation** — obstacle avoidance without range sensors
- **Autonomous driving** — dense depth priors for camera-only perception stacks
- **3D photography** — portrait mode, bokeh simulation, refocus

---

## Inference

```python
from depthpro import DepthProEngine

engine = DepthProEngine.load("checkpoints/dpt-finetuned.pt", device="cuda")

# Single image
depth_map = engine.predict("scene.jpg")   # returns (H, W) float32 metres
engine.visualize(depth_map, colormap="magma")

# Video stream
for frame, depth in engine.stream("video.mp4"):
    overlay = engine.overlay(frame, depth, alpha=0.6)
```

---

*Part of the [Sherif Abd El-Rady CV Portfolio](https://sherifabdelrady.replit.app)*
