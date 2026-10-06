# ABCDFSS Full-Dataset & Cloud Reproducibility Readiness Report

**Repository**: `KTD1108/ABCDFSS`  
**Git Branch**: `main`  
**Current HEAD**: `573f1ac` (and subsequent reproducibility enhancements)  
**Date**: October 6, 2026  
**Status**: AUDITED & BENCHMARK READY  

---

## 1. Executive Summary & Architecture Overview

The ABCDFSS (Adaptive Cross-Domain Few-Shot Segmentation) evaluation harness has been upgraded from ad-hoc episodic execution into an **auditable, deterministic, cryptographic-verified benchmark engine**.

### Architectural Baseline
All core algorithmic components remain strictly frozen:
- **Backbone**: ResNet-50 (`ResNet50_Weights.DEFAULT`), pre-trained on ImageNet.
- **Feature Extraction**: 16 feature layers extracted pre-ReLU (unclipped raw representations).
- **Intermediate Alignment Scale**: Spatial fusion anchored at layer $l_0 = 3$ ($50\times 50$ spatial grid for $400\times 400$ input).
- **Adaptation Mechanism**: Online SGD on support set for $E = 25$ epochs with learning rate $\eta = 10^{-2}$.
- **Thresholding Strategy**: Adaptive `pred_mean` threshold: $\tau = \max(\tau_{\text{Otsu}}, \mu_{\text{pred}})$, with `drop_least=0.05`.
- **Adaptation Protocol**: `every-episode` (CVPR 2024 Algorithm 2, adapting weights per evaluation episode).

### Frozen Experiment Matrix (E0 – E3)
| Experiment ID | Scientific Role | Adapter Architecture | Multi-Layer Fusion Strategy |
| :---: | :--- | :--- | :--- |
| **E0** | Original Baseline (CVPR 2024 reproduction) | Pointwise Conv $1\times 1$ | Uniform Mean Fusion |
| **E1** | Adapter Ablation | Depthwise Separable Conv $3\times 3$ | Uniform Mean Fusion |
| **E2** | Fusion Ablation | Pointwise Conv $1\times 1$ | Softmax Margin Fusion ($\tau=1.0$) |
| **E3** | Full Proposed Method | Depthwise Separable Conv $3\times 3$ | Softmax Margin Fusion ($\tau=1.0$) |

> [!IMPORTANT]
> The same episode manifest is strictly shared across E0, E1, E2, and E3 for every dataset, ensuring perfectly matched cross-experiment comparisons without sample variance.

---

## 2. Deterministic Manifest System

### Portable Relative Paths
Every episode manifest stores filepaths relative to the dataset base path (e.g. `ISIC2018_Task1-2_Training_Input/ISIC_0015559.jpg`), eliminating hardcoded Windows or absolute Linux paths. File paths are resolved dynamically at runtime by dataset loaders.

### Directory Convention
Standard manifests are located in:
```text
experiments/episodes/
    deepglobe_seed42_20episodes.json
    deepglobe_seed42_100episodes.json
    deepglobe_seed42_1000episodes.json
    deepglobe_seed42_all_episodes.json
    isic_seed42_...
    lung_seed42_...
    fss_seed42_...
    suim_seed42_...
```

### Automatic Resolution Flow
In benchmark mode, the runner automatically resolves the manifest:
$$\text{Manifest File} \xrightarrow{\quad} \texttt{run\_full\_benchmark\_suite.py} \xrightarrow{\text{--manifest}} \texttt{evaluate.py} \xrightarrow{\quad} \text{Dataset Loader} \xrightarrow{\quad} \text{E0--E3}$$

If a manifest does not exist, the runner **fails clearly** with an actionable error message rather than silently falling back to random sampling:
```text
[ERROR] No deterministic episode manifest found for benchmark run:
  Benchmark: deepglobe
  Seed:      42
  Episodes:  1000
  Expected:  D:\xulyanhv2\ABCDFSS\experiments\episodes\deepglobe_seed42_1000episodes.json

To generate it deterministically, run:
  python experiments/generate_manifests.py --benchmark deepglobe --episodes 1000 --seed 42
```

---

## 3. Clear Terminology & Evaluation Scales

To maintain scientific integrity, the following three terms are strictly distinguished:

1. **Fixed Evaluation Episodes (e.g., 20, 100, 1000 episodes)**:
   - A deterministic pseudo-random subset of evaluation episodes sampled under `seed=42`.
   - 1000 episodes matches the standard CVPR / CD-FSS literature convention (PAT, RT-FSS).
   - *It is NOT the full dataset.*
2. **All Episodes in Manifest (`--episodes all`)**:
   - The engine resolves `episodes = len(manifest["episodes"])` and evaluates every episode recorded in the manifest.
   - The manifest file itself serves as the authoritative, immutable specification of the evaluation set.
3. **Full Dataset**:
   - Exhaustive evaluation where every single image in the benchmark domain serves as a query at least once.

---

## 4. How to Generate Manifests

Use [`experiments/generate_manifests.py`](file:///d:/xulyanhv2/ABCDFSS/experiments/generate_manifests.py):

```bash
# Generate 20-episode manifests (standard local verification)
python experiments/generate_manifests.py --benchmark all --episodes 20 --seed 42

# Generate 100-episode manifests
python experiments/generate_manifests.py --benchmark isic --episodes 100 --seed 42

# Generate 1000-episode CVPR standard manifests for all 5 domains
python experiments/generate_manifests.py --benchmark all --episodes 1000 --seed 42

# Generate exhaustive "all" manifest for a domain
python experiments/generate_manifests.py --benchmark lung --episodes all --seed 42
```

---

## 5. How to Run Evaluations

### A. Running Standard 20 Episodes (Historical Baseline Verification)
```bash
# Runs E0-E3 across all 5 datasets; preserves historical results in results/full_benchmark/
python run_full_benchmark_suite.py --episodes 20 --seed 42 --device cuda
```

### B. Running 100 Episodes
```bash
# Outputs automatically route to results/full_benchmark_100ep/
python run_full_benchmark_suite.py --episodes 100 --seed 42 --device cuda
```

### C. Running 1000 Episodes (Standard Literature Benchmark)
```bash
# First generate manifests:
python experiments/generate_manifests.py --benchmark all --episodes 1000 --seed 42

# Run full suite (isolated in results/full_benchmark_1000ep/):
python run_full_benchmark_suite.py --episodes 1000 --seed 42 --device cuda --nworker 4
```

### D. Running All Episodes in Manifest (`--episodes all`)
```bash
# Evaluates every episode listed in the resolved manifest:
python run_full_benchmark_suite.py --episodes all --seed 42 --device cuda
```

### E. Running Single Experiment with `evaluate.py`
```bash
# Automatically resolves experiments/episodes/isic_seed42_20episodes.json:
python evaluate.py --benchmark isic --datapath /path/to/isic --experiment E0 --episodes 20 --device cuda
```

---

## 6. Strong Resume Validation

Previous naive checks only verified `episode_count == requested_episodes`. The updated pipeline enforces **cryptographic protocol signatures**:

```json
{
  "protocol_signature": {
    "benchmark": "deepglobe",
    "experiment": "E3",
    "episodes": 1000,
    "seed": 42,
    "nshot": 1,
    "adapt_to": "every-episode",
    "adapter": "depthwise_separable_3x3",
    "fusion": "softmax_margin",
    "fusion_temp": 1.0,
    "image_size": 400,
    "num_epochs": 25,
    "learning_rate": 0.01,
    "manifest": "/path/to/deepglobe_seed42_1000episodes.json",
    "manifest_sha256": "4a7f8e... (SHA-256 hash of manifest file)"
  }
}
```

### Resume Rules
- When inspecting an existing `run_result.json`, `validate_protocol_signature()` checks all 12 critical protocol fields.
- If **any** parameter differs (seed, adapter, fusion, learning rate, or manifest SHA-256), the existing result is **rejected** and the experiment is re-run.
- Skipping only occurs when the protocol signature matches 100% identically:
  `[SKIP - SIGNATURE VERIFIED] DeepGlobe E3 => Cumulative mIoU: 77.20%, Mean Episode-IoU: 77.15%`

---

## 7. Cloud GPU Execution via Modal

[`modal_runner.py`](file:///d:/xulyanhv2/ABCDFSS/modal_runner.py) executes serverless GPU jobs (T4 / A10G / A100) using the exact same code and manifests:

### Architecture
- **Persistent Volume 1** (`abcdfss-datasets-cache` $\rightarrow$ `/root/datasets_cache`):
  Caches Kaggle and Hugging Face datasets across container lifecycles (`KAGGLEHUB_CACHE=/root/datasets_cache/kagglehub`).
- **Persistent Volume 2** (`abcdfss-results` $\rightarrow$ `/root/results`):
  Persists all log files and structured `run_result.json` artifacts.
- **Manifest Synchronization**:
  Modal automatically resolves `/root/ABCDFSS/experiments/episodes/{benchmark}_seed{seed}_{episodes}episodes.json`, which matches the caller's local manifest bit-for-bit.
- **Telemetry Injection**:
  The caller's git commit SHA and status are injected into container environment variables (`ABCDFSS_GIT_COMMIT`, `ABCDFSS_GIT_DIRTY`) and recorded in the artifact.

### Modal Commands
```bash
# Run single experiment E0 on ISIC for 100 episodes on T4:
modal run modal_runner.py --benchmark isic --experiment E0 --episodes 100

# Run proposed E3 on DeepGlobe for 1000 episodes on A10G:
modal run modal_runner.py --benchmark deepglobe --experiment E3 --episodes 1000 --gpu A10G

# Run full matrix on Lung for 100 episodes:
modal run modal_runner.py --benchmark lung --experiment all --episodes 100
```

---

## 8. CPU vs. GPU Reproducibility Audit

To audit numerical consistency across heterogeneous compute devices (e.g. CPU vs. CUDA), use [`experiments/check_reproducibility.py`](file:///d:/xulyanhv2/ABCDFSS/experiments/check_reproducibility.py):

```bash
python experiments/check_reproducibility.py \
    --run1 results/run_cpu/run_result.json \
    --run2 results/run_gpu/run_result.json \
    --tolerance-mean 0.15 \
    --tolerance-max 0.50
```

### Tolerances
- **Mean Episode-IoU Tolerance**: $\le 0.15\%$
- **Cumulative mIoU Tolerance**: $\le 0.15\%$
- **Single-Episode Max Delta**: $\le 0.50\%$

*Explanation of Differences*: CPU floating-point accumulation (sequential MKL/OpenBLAS) versus GPU tensor core reductions (parallel warp reductions and non-deterministic atomic adds in CUDA backward passes) introduce minor rounding differences at $10^{-5}$ precision, propagating to small fractional differences in thresholding boundaries. The specified tolerances ensure scientific protocol equivalence without requiring bitwise identity.

---

## 9. Comprehensive Environment Inventory

| Component | Local Audit Environment | Modal GPU Cloud Environment |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) | Debian GNU/Linux 12 (bookworm) |
| **Python** | 3.14.3 | 3.10.14 |
| **PyTorch** | 2.14.0+cpu | 2.1.2+cu121 |
| **TorchVision** | 0.29.0+cpu | 0.16.2+cu121 |
| **NumPy** | 2.5.2 | 1.26.4 (`numpy<2.0.0`) |
| **Pillow** | 12.3.0 | 10.2.0 |
| **CUDA** | N/A (CPU Only) | CUDA 12.1 / cuDNN 8.9 |
| **Default Accelerators** | Intel Core / AMD CPU | NVIDIA Tesla T4 (16 GB) / A10G (24 GB) |

Every `run_result.json` records complete environment telemetry under `"environment"`.

---

## 10. Enriched Granular Episode Artifacts

In addition to aggregate metrics, `run_result.json` now includes traceable per-episode records:

```json
{
  "raw_result": {
    "episode_ious": [78.52, 69.41, 84.10],
    "detailed_episodes": [
      {
        "episode_id": 0,
        "class_id": 0,
        "category": "1",
        "query_img": "ISIC2018_Task1-2_Training_Input/ISIC_0015559.jpg",
        "support_imgs": ["ISIC2018_Task1-2_Training_Input/ISIC_0015274.jpg"],
        "iou": 78.5213
      }
    ]
  }
}
```
Existing result readers that expect `episode_ious` as a list of floats remain 100% compatible.

---

## 11. Known Limitations & Best Practices

1. **Local Windows Execution Speed**:
   Online episodic SGD on CPU takes $\approx 35\text{--}90$ seconds per episode. For $N=1000$ episodes, local CPU evaluation requires $\approx 15\text{--}25$ hours per dataset. Use Modal GPU for large-scale benchmarks.
2. **KaggleHub Download Quotas**:
   Downloading multi-gigabyte datasets concurrently can hit Kaggle API rate limits. The Modal runner caches datasets in persistent storage (`abcdfss-datasets-cache`) so download occurs only once per dataset.
3. **Dataset Manifest Verification**:
   Always verify manifest integrity with `validate_manifest()` or run unit tests (`python -m unittest discover -s tests`) prior to launching cloud GPU suites.
