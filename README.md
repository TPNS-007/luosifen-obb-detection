#  luosifen-obb-detection
Lightweight oriented detection for Luosifen packaging on edge devices. Dataset: 1,603 OBB images (4 classes, 0°–360°). Model: improved YOLOv11n-OBB with GhostConv_OBB + AAE + PECA. Achieves 93.95% mAP@0.5:0.95 at 2.95M params / 14.76 FPS.



# Luosifen OBB Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Lightweight oriented detection for Luosifen packaging on edge devices.**  
Dataset: 1,603 OBB images (4 classes, 0°–360°). Model: improved YOLOv11n-OBB with GhostConv_OBB, AAE and PECA. Achieves **93.95% mAP@0.5:0.95** at **2.95 M params** / **14.76 FPS**.

This is the official implementation of *"Research on the Lightweight Recognition Method of Luosifen Outer Packaging Based on Improved YOLOv11n-OBB"*.

---

## 📦 What's Included

- **Luosifen Outer Packaging OBB Dataset**  
  1,603 images with oriented bounding box annotations (8-parameter format), covering 4 commercial varieties under stationary and dynamic conveyor-belt conditions.
- **Improved YOLOv11n-OBB Source Code**  
  GhostConv_OBB, C3k2_GhostConv_OBB, Angle-Aware Enhancement (AAE), Progressive Efficient Channel Attention (PECA), and safe fallback mechanisms.
- **Training & Evaluation Scripts**  
  Reproducible configs and pretrained weights for edge-side deployment.

---

## 📊 Dataset

| Attribute | Details |
|:---|:---|
| Categories | 4 varieties (A–D) |
| Total Images | 1,603 |
| Annotation Format | YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized) |
| Angle Range | 0°–360° |
| Data Split | Train 1,122 (70%) / Val 320 (20%) / Test 161 (10%) |
| Acquisition | Simulated boxing-machine conveyor belt (stationary + dynamic) |

---

## 🏗️ Model Architecture

| Module | Description |
|:---|:---|
| **GhostConv_OBB** | Lightweight feature extraction with cheap operations and learnable residual fusion |
| **AAE** | Dual-path Angle-Aware Enhancement (global semantic + local geometric) for edge/corner features |
| **PECA** | Progressive Efficient Channel Attention (two-stage 1D conv) for background suppression |
| **C3k2_GhostConv_OBB** | Multi-scale fusion module deployed at the **Neck-to-Head transition layer (Head 11)** via key-layer sensitivity analysis |
| **Safe Fallback** | Automatic identity-mapping fallback on NaN/Inf for edge-device stability |

---
