<div align="center">

# ABCDFSS: Khảo Sát Kiến Trúc & Tái Lập Khoa Học Cho Phân Đoạn Ảnh Miền Chéo Few-Shot (CD-FSS)
### Adapt Before Comparison: Architectural Ablation and Rigorous Reproducibility Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Modal GPU](https://img.shields.io/badge/Modal-Tesla%20T4%20GPU-00C49F.svg)](https://modal.com/)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-34%2F34%20Passing-brightgreen.svg)]()
[![Seed Freeze](https://img.shields.io/badge/Seed-42%20Deterministic-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Khung nghiên cứu độc lập, chuẩn hóa 100% bằng PyTorch, tái lập trung thực thuật toán thích nghi online test-time SGD (CVPR 2024), cung cấp bộ đối chuẩn 20 thực nghiệm (E0–E3 trên 5 datasets) với manifest tất định và chữ ký giao thức mã hóa.*

---

</div>

## 📖 Mục Lục (Table of Contents)
1. [Giới Thiệu Đề Tài & Bối Cảnh Nghiên Cứu](#-1-giới-thiệu-đề-tài--bối-cảnh-nghiên-cứu)
2. [Thiết Kế Kiến Trúc & 4 Cấu Hình Ablation (E0–E3)](#-2-thiết-kế-kiến-trúc--4-cấu-hình-ablation-e0e3)
3. [Kết Quả Đối Chuẩn Toàn Diện 100-Episode GPU (Official Benchmark)](#-3-kết-quả-đối-chuẩn-toàn-diện-100-episode-gpu-official-benchmark)
4. [Phân Tích Khoa Học & Đánh Giá Giả Thuyết](#-4-phân-tích-khoa-học--đánh-giá-giả-thuyết)
5. [Cấu Trúc Thư Mục Dự Án](#-5-cấu-trúc-thư-mục-dự-án)
6. [Hướng Dẫn Tái Lập 1-Click (Quickstart & Reproduction)](#-6-hướng-dẫn-tái-lập-1-click-quickstart--reproduction)
7. [Bảo Chứng Tính Tái Lập & Nguồn Gốc Thực Thi (Provenance)](#-7-bảo-chứng-tính-tái-lập--nguồn-gốc-thực-thi-provenance)
8. [Tài Liệu Tham Khảo & Trích Dẫn](#-8-tài-liệu-tham-khảo--trích-dẫn)

---

## 📌 1. Giới Thiệu Đề Tài & Bối Cảnh Nghiên Cứu

Phân đoạn ngữ nghĩa ít mẫu miền chéo (**Cross-Domain Few-Shot Semantic Segmentation - CD-FSS**) giải quyết bài toán nhận diện và phân đoạn đối tượng trong các miền ảnh hoàn toàn mới (ảnh vệ tinh viễn thám, ảnh nội soi da liễu y tế, X-quang phổi, ảnh sinh vật dưới nước) khi chỉ có $K=1$ ảnh mẫu hỗ trợ (1-shot support image).

Bài báo gốc *"Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation"* (**CVPR 2024**, Heyou et al.) đề xuất phương pháp thích nghi trực tuyến tại thời điểm kiểm thử (**Test-Time Online SGD Adaptation**): đóng băng backbone ResNet-50 và chỉ cập nhật các adapter gắn vào từng tầng đặc trưng thông qua 25 epochs tối ưu hóa hàm mất mát tương phản liên tầng (Dense InfoNCE & Variance Loss) trên duy nhất 1 ảnh support.

### Động Lực Nghiên Cứu (Research Motivations)
1. **Khảo sát kernel không gian (Spatial Receptive Field):** Bài báo gốc thử nghiệm thay adapter Conv $1 \times 1$ bằng Standard Conv $3 \times 3$ nhưng thất bại do bùng nổ tham số ($>1.2\text{M}$ tham số) gây quá khớp nặng trên 1 ảnh mẫu. Chúng tôi đề xuất giải pháp **Depthwise Separable Conv $3 \times 3$ với Residual Shortcut** nhằm mở rộng receptive field không gian mà vẫn giữ số tham số ở mức tối thiểu.
2. **Khảo sát cơ chế gộp tầng (Layer Fusion Mechanism):** Thay vì gộp trung bình phẳng cào bằng ($\frac{1}{L} \sum \hat{q}^l$), chúng tôi khảo sát cơ chế **Softmax Margin Fusion** tự động cấp trọng số dựa trên độ phân biệt prototype tiền cảnh/hậu cảnh của từng tầng.
3. **Tính trung thực khoa học & Chuẩn hóa thực nghiệm:** Xây dựng lại toàn bộ pipeline độc lập, loại bỏ code rườm rà, cố định seed 42 với manifest JSON lưu hash SHA-256 cho từng episode để đảm bảo mọi so sánh E0–E3 đều tuyệt đối công bằng trên cùng tập dữ liệu.

---

## 💡 2. Thiết Kế Kiến Trúc & 4 Cấu Hình Ablation (E0–E3)

```
[Query & Support Images] (400x400)
       │
       ▼
[ResNet-50 Frozen Backbone] ──> 13 Trọng số tiền kích hoạt Pre-ReLU
       │
       ▼
[Adapter Modules] ────────────> (Conv 1x1 HOẶC Depthwise Separable 3x3)
       │
       ▼
[Dense Cross-Attention] ──────> Ma trận tương quan đa tầng Q_l
       │
       ▼
[Layer Fusion] ───────────────> (Mean Fusion HOẶC Softmax Margin Fusion)
       │
       ▼
[Adaptive Thresholding] ──────> max(Otsu, mean), drop_least=0.05 ──> Mặt nạ phân đoạn Query
```

### Ma Trận Định Nghĩa Thực Nghiệm

| Mã | Tên Cấu Hình | Adapter Module | Layer Fusion | Vai Trò Khoa Học |
| :---: | :--- | :--- | :--- | :--- |
| **E0** | **Baseline** | Conv $1 \times 1$ Pointwise | Mean Fusion | Tái lập trung thực kiến trúc gốc ABCDFSS (CVPR 2024) |
| **E1** | **Adapter Ablation** | Depthwise Separable $3 \times 3$ | Mean Fusion | Cô lập tác động của việc mở rộng receptive field không gian |
| **E2** | **Fusion Ablation** | Conv $1 \times 1$ Pointwise | Softmax Margin Fusion | Cô lập tác động của cơ chế gộp tầng phân tách miền |
| **E3** | **Full Proposed** | Depthwise Separable $3 \times 3$ | Softmax Margin Fusion | Đánh giá hiệu năng kết hợp cả hai module cải tiến |

---

## 🏆 3. Kết Quả Đối Chuẩn Toàn Diện 100-Episode GPU (Official Benchmark)

Toàn bộ **20 cấu hình thực nghiệm** (5 datasets $\times$ 4 experiments E0–E3) được đánh giá trên hệ thống **Modal Cloud GPU (NVIDIA Tesla T4 16GB)** với **100 episodes tất định** được kiểm soát chặt chẽ qua manifest (tổng cộng **2.000 episodes**).

### 3.1. Bảng Kết Quả Cumulative mIoU (%)

> **Cumulative mIoU**: Tích lũy tổng diện tích giao (Intersection) và hợp (Union) trên toàn bộ các pixel query của từng lớp trước khi tính trung bình macro:
> $$\text{IoU}_c = \frac{\sum_{i \in \mathcal{E}_c} |P_i \cap G_i|}{\sum_{i \in \mathcal{E}_c} |P_i \cup G_i|}, \quad \text{Cumulative mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c$$

| Tập Dữ Liệu (Domain) | Số Lớp | E0 (Base) | E1 (Adp) | $\Delta$ E1 | E2 (Fus) | $\Delta$ E2 | E3 (Prop) | $\Delta$ E3 vs E0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** *(Viễn thám)* | 6 | **44.74%** | 44.08% | -0.66 pp | 44.44% | -0.30 pp | 43.73% | -1.01 pp |
| **ISIC 2018** *(Da liễu)* | 3 | **43.29%** | 41.80% | -1.49 pp | 42.68% | -0.61 pp | 41.72% | -1.57 pp |
| **Lung / CXR** *(X-quang ngực)* | 1 (Binary) | 81.32% | 81.90% | +0.58 pp | 81.16% | -0.16 pp | **82.21%** | **+0.89 pp** 🚀 |
| **FSS-1000** *(Ảnh tự nhiên)* | 240 | 69.85% | 65.73% | -4.12 pp | **69.89%** | +0.04 pp | 66.49% | -3.36 pp |
| **SUIM** *(Ảnh dưới nước)* | 5 | **37.01%** | 35.16% | -1.85 pp | 36.68% | -0.33 pp | 35.35% | -1.66 pp |
| **TRUNG BÌNH SUITE** | — | **55.24%** | **53.73%** | **-1.51 pp** | **54.97%** | **-0.27 pp** | **53.90%** | **-1.34 pp** |

---

### 3.2. Bảng Kết Quả Mean Episode-IoU (%)

> **Mean Episode-IoU**: Trung bình số học không trọng số của IoU từng episode:
> $$\text{Mean Episode-IoU} = \frac{1}{N} \sum_{i=1}^N \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$

| Tập Dữ Liệu (Domain) | Số Lớp | E0 (Base) | E1 (Adp) | $\Delta$ E1 | E2 (Fus) | $\Delta$ E2 | E3 (Prop) | $\Delta$ E3 vs E0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** *(Viễn thám)* | 6 | **44.06%** | 43.85% | -0.21 pp | 43.77% | -0.29 pp | 43.30% | -0.76 pp |
| **ISIC 2018** *(Da liễu)* | 3 | **47.07%** | 45.57% | -1.50 pp | 46.61% | -0.46 pp | 45.37% | -1.70 pp |
| **Lung / CXR** *(X-quang ngực)* | 1 (Binary) | 81.31% | 81.74% | +0.43 pp | 81.12% | -0.19 pp | **82.08%** | **+0.77 pp** 🚀 |
| **FSS-1000** *(Ảnh tự nhiên)* | 240 | 69.85% | 65.73% | -4.12 pp | **69.89%** | +0.04 pp | 66.49% | -3.36 pp |
| **SUIM** *(Ảnh dưới nước)* | 5 | **39.56%** | 35.75% | -3.81 pp | 39.35% | -0.21 pp | 35.88% | -3.68 pp |
| **TRUNG BÌNH SUITE** | — | **56.37%** | **54.53%** | **-1.84 pp** | **56.15%** | **-0.22 pp** | **54.62%** | **-1.75 pp** |

---

### 3.3. Đối Chiếu Tham Khảo Với Số Liệu Bài Báo Gốc (CVPR 2024)

> [!NOTE]
> Số liệu bài báo gốc được trích dẫn từ bài báo CVPR 2024 (chạy trên 1.000 random episodes). Bảng dưới đây đóng vai trò tham chiếu độ khó của từng miền dữ liệu, **không phải là mục tiêu đối đầu trực tiếp** vì khác biệt quy trình (1.000 random episodes ở bài báo gốc vs 100 episodes tất định cố định theo manifest ở đây).

| Miền Dữ Liệu | Số Liệu Tham Khảo Bài Báo (CVPR 2024) | E0 (Base Tái Lập) | E1 (Adapter) | E2 (Fusion) | E3 (Kết Hợp) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 42.3% | 44.74% | 44.08% | 44.44% | 43.73% |
| **ISIC 2018** | 41.8% | 43.29% | 41.80% | 42.68% | 41.72% |
| **Lung / CXR** | 80.0% | 81.32% | 81.90% | 81.16% | **82.21%** |
| **FSS-1000** | 69.3% | 69.85% | 65.73% | 69.89% | 66.49% |
| **SUIM** | 35.0% | 37.01% | 35.16% | 36.68% | 35.35% |

---

## 🔬 4. Phân Tích Khoa Học & Đánh Giá Giả Thuyết

Từ kết quả thực nghiệm 20 runs, chúng tôi rút ra 4 kết luận khoa học cốt lõi:

1. **Hiệu năng của Adapter $3 \times 3$ phụ thuộc mạnh vào miền dữ liệu (Domain Dependency):**
   - Trên **Lung (CXR)**: Cấu trúc giải phẫu vòm hoành và lồng ngực mang tính liên tục không gian cao. Receptive field $3 \times 3$ giúp gom biên tốt hơn, tăng **+0.58 pp** (Cumulative mIoU) và **+0.43 pp** (Mean Episode-IoU).
   - Trên các miền đa lớp (ISIC, FSS-1000, SUIM): Việc thêm trọng số không gian $3 \times 3$ trong điều kiện chỉ học 25 epochs trên 1 ảnh mẫu duy nhất gây phân tán gradient so với phép chiếu $1 \times 1$ pointwise trực tiếp, khiến điểm số giảm từ 0.66 đến 4.12 pp.
2. **Cơ chế Softmax Margin Fusion có độ ổn định cao:**
   - Softmax Margin Fusion bám sát rất chặt baseline E0 (chênh lệch chỉ -0.16 pp đến -0.61 pp trên 4 datasets, và nhích nhẹ +0.04 pp trên FSS-1000). Cơ chế phân chia trọng số dựa trên prototype margin là một giải pháp an toàn, không gây sụp đổ biểu diễn.
3. **Hiệu ứng tương tác phi tuyến (Super-Additive Interaction) ở E3:**
   - Phân tích tương tác $\Delta_{Combined} - (\Delta_{Adapter} + \Delta_{Fusion})$ cho thấy giá trị dương trên 4/5 dataset: **+0.47 pp** (Lung), **+0.72 pp** (FSS-1000), **+0.53 pp** (ISIC), **+0.52 pp** (SUIM). Điều này chứng minh rằng việc kết hợp Adapter và Fusion tạo ra sự tương hỗ phi tuyến, không phải phép cộng rời rạc.
4. **Tính minh bạch và trung thực học thuật:**
   - E3 đạt **82.21%** trên Lung (tăng +0.89 pp so với E0). Tuy nhiên, dự án không tuyên bố đơn giản là "vượt paper" vì giao thức đánh giá khác biệt (100 deterministic episodes vs 1.000 random episodes). Trên trung bình toàn suite, E0 vẫn là baseline cực kỳ mạnh mẽ và tối ưu nhất cho bài toán thích nghi 1-shot trực tuyến.

---

## 📁 5. Cấu Trúc Thư Mục Dự Án

Mã nguồn được tổ chức tinh gọn, chuẩn mực tương tự cấu trúc tác giả gốc nhưng áp dụng mô-đun hóa hiện đại:

```
d:/xulyanhv2/ABCDFSS/
├── src/                               # Toàn bộ mã nguồn cốt lõi (Core Engine & Models)
│   ├── datasets/                      # Dataloaders cho DeepGlobe, ISIC, Lung, FSS-1000, SUIM
│   │   ├── builder.py                 # Hàm dựng dataloader chung
│   │   ├── deepglobe.py               # Xử lý ảnh viễn thám đa lớp
│   │   ├── isic.py                    # Xử lý ảnh da liễu ISIC-2018
│   │   ├── lung.py                    # Xử lý X-quang phổi nhị phân
│   │   ├── fss.py                     # Xử lý 240 lớp FSS-1000
│   │   └── suim.py                    # Xử lý 5 lớp ảnh dưới nước
│   ├── engine/
│   │   └── pipeline.py                # CD-FSS Engine thích nghi trực tuyến (Algorithm 2)
│   ├── metrics/
│   │   ├── metrics.py                 # MetricTracker (Cumulative mIoU & Mean Ep-IoU)
│   │   └── thresholding.py            # Phân ngưỡng nhị phân Otsu & Mean
│   ├── models/
│   │   ├── adapters.py                # Factory dựng Adapter 1x1 và Depthwise 3x3
│   │   ├── attention.py               # Dense Cross-Attention
│   │   ├── backbone.py                # ResNet-50 Feature Extractor
│   │   ├── fusion.py                  # Factory dựng Mean Fusion & Softmax Margin Fusion
│   │   └── loss.py                    # Dense InfoNCE Loss & Keep Variance Loss
│   └── utils/
│       ├── augmentations.py           # Phép biến đổi tăng cường dữ liệu support
│       ├── manifest.py                # Trình phân giải & thẩm định manifest (SHA256)
│       └── protocol.py                # Chữ ký giao thức thực nghiệm & resume
│
├── experiments/                       # Quản lý thực nghiệm & tái lập
│   ├── episodes/                      # Tập JSON manifest tất định (20ep và 100ep cho 5 datasets)
│   ├── check_reproducibility.py       # Công cụ kiểm tra sai số số học & tính tái lập
│   └── generate_manifests.py          # Script sinh manifest tất định từ seed
│
├── results/                           # Kết quả benchmark 100-episode chính thức
│   ├── deepglobe/                     # Kết quả 100ep E0-E3 DeepGlobe (JSON + logs)
│   ├── isic/                          # Kết quả 100ep E0-E3 ISIC (JSON + logs)
│   ├── lung/                          # Kết quả 100ep E0-E3 Lung (JSON + logs)
│   ├── fss/                           # Kết quả 100ep E0-E3 FSS-1000 (JSON + logs)
│   └── suim/                          # Kết quả 100ep E0-E3 SUIM (JSON + logs)
│
├── tests/                             # Bộ kiểm thử đơn vị (34 unit tests hoàn chỉnh)
│   ├── test_pipeline.py               # Test kiến trúc, loss, adapters, thresholding
│   └── test_reproducibility.py        # Test manifest, protocol signature, sai số
│
├── docs/                              # Tài liệu báo cáo nghiên cứu & đặc tả môi trường
│   ├── BASELINE_FREEZE_REPORT.md      # Báo cáo đóng băng baseline E0 chính thức
│   ├── FULL_BENCHMARK_REPORT.md       # Báo cáo tổng hợp khoa học 20 runs chi tiết
│   └── ENVIRONMENT.md                 # Đặc tả môi trường phần cứng, phần mềm, CUDA
│
├── scripts/                           # Công cụ phụ trợ tổng hợp báo cáo
│   └── generate_full_benchmark_report.py # Script tổng hợp báo cáo FULL_BENCHMARK_REPORT.md
│
├── main.py                            # Điểm vào chuẩn tác giả gốc CVPR 2024
├── evaluate.py                        # Điểm vào thực thi chi tiết (Single-run CLI)
├── run_all_benchmarks.py              # Master runner điều phối toàn bộ suite 20 runs
├── modal_runner.py                    # Runner không máy chủ trên Modal Cloud GPU (Tesla T4)
├── requirements.txt                   # Danh sách gói phụ thuộc Python
└── README.md                          # Tài liệu hướng dẫn chính của dự án
```

---

## 🚀 6. Hướng Dẫn Tái Lập 1-Click (Quickstart & Reproduction)

### 6.1. Cài Đặt Môi Trường (Installation)

```bash
git clone https://github.com/KTD1108/ABCDFSS.git
cd ABCDFSS
pip install -r requirements.txt
```

### 6.2. Chạy Bộ Kiểm Thử Đơn Vị (Run Unit Tests)

Để xác nhận hệ thống hoạt động chính xác trước khi thực thi thực nghiệm:

```bash
python -m unittest discover tests
# Kết quả mong đợi: Ran 34 tests in ~28s — OK
```

### 6.3. Chạy Một Thực Nghiệm Đơn Lẻ (Single Benchmark)

Dự án hỗ trợ 2 cách gọi:

```bash
# Cách 1: Chuẩn giao diện bài báo gốc tác giả CVPR 2024 (main.py)
python main.py --benchmark isic --datapath ./datasets/isic --nshot 1

# Cách 2: Mở rộng với cấu hình thử nghiệm E0-E3 và episodes manifest (main.py hoặc evaluate.py)
python main.py --benchmark isic --experiment E3 --episodes 100 --seed 42 --device cuda
```

### 6.4. Chạy Toàn Bộ Bộ Đối Chuẩn (Full 20-Run Suite Locally)

Chạy tất cả 4 thực nghiệm (E0–E3) trên cả 5 datasets (tự động tải dữ liệu nếu chưa có, tự động khóa manifest và xác thực chữ ký giao thức):

```bash
python run_all_benchmarks.py \
    --experiments E0 E1 E2 E3 \
    --episodes 100 \
    --seed 42 \
    --device cuda
```

### 6.5. Chạy Trên Đám Mây Modal Serverless GPU (Cloud GPU Execution)

Nếu muốn chạy trên GPU đám mây Tesla T4 / A10G thông qua [Modal](https://modal.com/):

```bash
# 1. Cài đặt và cấu hình token Modal
pip install modal
modal setup

# 2. Chạy 1 thực nghiệm trên Cloud GPU
modal run modal_runner.py --benchmark isic --experiment E3 --episodes 100

# 3. Chạy toàn bộ 20 thực nghiệm song song trên Cloud GPU
modal run modal_runner.py --benchmark all --experiment all --episodes 100
```

### 6.6. Kiểm Tra Tính Tái Lập Số Học (Audit Cross-Device Reproducibility)

Để thẩm định tính nhất quán số học giữa 2 tệp kết quả với ngưỡng sai số nghiêm ngặt ($\Delta_{\text{mean}} \le 0.15\%$, $\Delta_{\text{max}} \le 0.50\%$):

```bash
python experiments/check_reproducibility.py \
    --run1 results/isic/E0_100ep_seed42/run_result.json \
    --run2 results/isic/E1_100ep_seed42/run_result.json \
    --tolerance-mean 0.15 \
    --tolerance-max 0.50
```

### 6.7. Tái Tạo Báo Cáo Đối Chuẩn (Generate Official Benchmark Report)

```bash
python scripts/generate_full_benchmark_report.py
# Cập nhật trực tiếp kết quả vào docs/FULL_BENCHMARK_REPORT.md
```

---

## 🔒 7. Bảo Chứng Tính Tái Lập & Nguồn Gốc Thực Thi (Provenance)

Để đảm bảo tính minh bạch khoa học tuyệt đối, toàn bộ dữ liệu thực nghiệm được lưu vết:

| Thông Tin Kiểm Toán | Giá Trị Cố Định |
| :--- | :--- |
| **Commit Mã Nguồn Thực Thi (Code Commit)** | [`0ad471e`](https://github.com/KTD1108/ABCDFSS/commit/0ad471e) *(container patch [`6575747`](https://github.com/KTD1108/ABCDFSS/commit/6575747))* |
| **Commit Lưu Trữ Kết Quả (Results Commit)** | [`635022c`](https://github.com/KTD1108/ABCDFSS/commit/635022c) / [`ca7bd23`](https://github.com/KTD1108/ABCDFSS/commit/ca7bd23) |
| **Random Seed** | `seed=42` cố định tuyệt đối |
| **Tất Định Manifest (Manifest Determinism)** | 100% episodes được lưu thành file JSON trong `experiments/episodes/`, thẩm định bằng SHA-256 |
| **Chữ Ký Giao Thức (Protocol Signature)** | Mỗi tệp `run_result.json` chứa chữ ký mã hóa gồm 13 trường siêu tham số nhằm chống nhầm lẫn dữ liệu |
| **Môi Trường Tính Toán (Hardware Environment)** | NVIDIA Tesla T4 GPU (16GB VRAM), Debian Slim, PyTorch 2.1.2 + CUDA 12.1 |

Chi tiết toàn văn các báo cáo thẩm định được lưu trữ tại:
- 📄 [Báo Cáo Đóng Băng Baseline (docs/BASELINE_FREEZE_REPORT.md)](docs/BASELINE_FREEZE_REPORT.md)
- 📄 [Báo Cáo Đối Chuẩn Đầy Đủ (docs/FULL_BENCHMARK_REPORT.md)](docs/FULL_BENCHMARK_REPORT.md)
- 📄 [Đặc Tả Môi Trường Kỹ Thuật (docs/ENVIRONMENT.md)](docs/ENVIRONMENT.md)

---

## 📚 8. Tài Liệu Tham Khảo & Trích Dẫn

```bibtex
@inproceedings{heyou2024adapt,
  title={Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation},
  author={Heyou, Jue and Zhang, Chi and Ding, Henghui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3432--3442},
  year={2024}
}
```

---

<div align="center">
  <b>ABCDFSS Research Suite & Benchmark</b> • Phát triển và hoàn thiện độc lập phục vụ nghiên cứu thị giác máy tính CD-FSS.
</div>
