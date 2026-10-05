# COMPUTATIONAL ENVIRONMENT SPECIFICATION

This document details the exact execution environment used for the baseline reproduction and controlled experiments of ABCDFSS (CVPR 2024).

---

## 1. System Hardware & Operating System

- **Operating System**: Microsoft Windows 11 Home / Pro (x86_64 / AMD64)
- **Host Architecture**: AMD64 (64-bit)
- **Hardware Acceleration**: CPU (Execution performed locally on CPU; Modal GPU cluster configured via `modal_runner.py` for cloud scale)
- **CUDA Device**: None (CPU execution)
- **Thread Parallelism**: PyTorch OpenMP default

---

## 2. Python Runtime & Core Libraries

- **Python Interpreter**: `3.14.3` (tags/v3.14.3:323c59a, Feb 3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]
- **PyTorch (`torch`)**: `2.14.0+cpu`
- **TorchVision (`torchvision`)**: `0.29.0+cpu`
- **NumPy (`numpy`)**: `2.5.2`
- **OpenCV (`opencv-python` / `cv2`)**: `5.0.0`
- **Pillow (`PIL`)**: `12.3.0`
- **TensorboardX (`tensorboardX`)**: `2.6.5`

---

## 3. Environment Reproducibility Verification

```bash
python -c "import sys, torch, torchvision, numpy, cv2; print('Python:', sys.version); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected Output:
```text
Python: 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]
Torch: 2.14.0+cpu
Torchvision: 0.29.0+cpu
NumPy: 2.5.2
OpenCV: 5.0.0
CUDA available: False
```

---

## 4. Episode Manifest Standard & Evaluation Protocol Freeze

- **Canonical Manifests**: To eliminate cross-platform pseudo-random number generator divergence, all experimental evaluations (E0, E1, E2, E3) are evaluated against fixed episode manifests stored under `experiments/episodes/`.
- **Default Adaptation Mode**: Frozen to `--adapt-to every-episode` (CVPR 2024 Algorithm 2). Quick-infer `--adapt-to first-episode` is retained as an optional speedup mode.
- **Metric Standard**: Pascal VOC / Cumulative mIoU ($\frac{\sum I}{\sum U}$) is logged alongside per-episode Mean IoU ($\frac{1}{N}\sum \frac{I}{U}$).
- **Reproduction Guarantee**: Metric Equivalence ($\Delta \le 0.01\text{ pp}$ on identical metric).

