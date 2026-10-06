# FULL BENCHMARK REPORT: E0–E3 EVALUATION ACROSS 5 DATASETS

## Đề Tài: Cross-Domain Few-Shot Semantic Segmentation (CD-FSS)
### Khảo Sát Độc Lập Ablation Study: Original ABCDFSS Baseline vs. Proposed Architecture

- **Execution Commit (Mã nguồn thực thi)**: [`0ad471e`](https://github.com/KTD1108/ABCDFSS/commit/0ad471e) (với runtime container patch [`6575747`](https://github.com/KTD1108/ABCDFSS/commit/6575747))
- **Results & Report Commit (Lưu trữ kết quả)**: [`635022c`](https://github.com/KTD1108/ABCDFSS/commit/635022c)

---

## 1. Experimental Protocol (Quy Trình Thực Nghiệm Chuẩn Hóa)

Toàn bộ 20 thử nghiệm (5 datasets × 4 cấu hình) được thực thi nghiêm ngặt theo đúng protocol cố định trên Modal Cloud GPU (Tesla T4):
Each configuration was evaluated on 100 deterministic seed-controlled episodes per dataset using explicit episode manifests generated under seed=42.

```yaml
seed: 42
nshot: 1
image_size: 400x400
adapt_to: every-episode (Algorithm 2: Test-Time Online SGD, 25 epochs)
learning_rate: 0.01
out_channels: 64
l0: 3 (Intermediate resolution 50x50)
threshold: max(Otsu, mean) with drop_least=0.05
episodes_per_run: 100 deterministic episodes (100% manifest-backed across all 5 datasets)
hardware: Modal Cloud GPU (NVIDIA Tesla T4, 16GB VRAM)
execution_commit: 0ad471e (fss patch: 6575747)
results_commit: 635022c
backbone: ResNet-50 (Pre-ReLU unclipped features, ImageNet weights frozen)
```

## 2. Experiment Matrix Definition

| Experiment | Adapter Architecture | Layer Fusion Mechanism | Role |
| :---: | :--- | :--- | :--- |
| **E0** | Conv $1\times 1$ (Pointwise) | Mean Fusion | Original ABCDFSS Baseline |
| **E1** | Depthwise Separable Conv $3\times 3$ | Mean Fusion | Adapter Ablation (Isolating Adapter) |
| **E2** | Conv $1\times 1$ (Pointwise) | Softmax Margin Fusion | Fusion Ablation (Isolating Fusion) |
| **E3** | Depthwise Separable Conv $3\times 3$ | Softmax Margin Fusion | Full Proposed Method (Combined) |

## 3. Benchmark Results — Cumulative mIoU (%)

Cumulative mIoU denotes the mean IoU computed after aggregating class-wise intersection and union statistics across the evaluated episodes:
$$\text{IoU}_c = \frac{\sum_{i \in \mathcal{E}_c} |P_i \cap G_i|}{\sum_{i \in \mathcal{E}_c} |P_i \cup G_i|}, \quad \text{Cumulative mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c$$

> [!NOTE]
> For the binary/single-class target domain (Lung), this corresponds to the aggregated foreground ratio $\frac{\sum |P_i \cap G_i|}{\sum |P_i \cup G_i|}$. For multi-class benchmarks (DeepGlobe, ISIC, FSS-1000, SUIM), intersection and union are aggregated per semantic class $c$ before computing the macro-average.

| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adapter | E2 (Fus) | $\Delta$ Fusion | E3 (Prop) | $\Delta$ Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 44.74% | 44.08% | -0.66 pp | 44.44% | -0.30 pp | 43.73% | -1.01 pp |
| **ISIC** | 43.29% | 41.80% | -1.49 pp | 42.68% | -0.61 pp | 41.72% | -1.57 pp |
| **Lung** | 81.32% | 81.90% | +0.58 pp | 81.16% | -0.16 pp | 82.21% | +0.89 pp |
| **FSS1000** | 69.85% | 65.73% | -4.12 pp | 69.89% | +0.04 pp | 66.49% | -3.36 pp |
| **SUIM** | 37.01% | 35.16% | -1.85 pp | 36.68% | -0.33 pp | 35.35% | -1.66 pp |
| **AVERAGE** | **55.24%** | **53.73%** | **-1.51 pp** | **54.97%** | **-0.27 pp** | **53.90%** | **-1.34 pp** |

---

## 4. Benchmark Results — Mean Episode-IoU (%)

Mean Episode-IoU denotes the unweighted average of individual episode IoU values:
$$\text{Mean Episode-IoU} = \frac{1}{N} \sum_{i=1}^N \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$

*(Reference Foreground-Background IoU metric)*:
$$FB\text{-IoU} = \frac{1}{2} \left( \frac{\sum_{i=1}^N |P_{fg, i} \cap G_{fg, i}|}{\sum_{i=1}^N |P_{fg, i} \cup G_{fg, i}|} + \frac{\sum_{i=1}^N |P_{bg, i} \cap G_{bg, i}|}{\sum_{i=1}^N |P_{bg, i} \cup G_{bg, i}|} \right)$$

| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adapter | E2 (Fus) | $\Delta$ Fusion | E3 (Prop) | $\Delta$ Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 44.06% | 43.85% | -0.21 pp | 43.77% | -0.29 pp | 43.30% | -0.76 pp |
| **ISIC** | 47.07% | 45.57% | -1.50 pp | 46.61% | -0.46 pp | 45.37% | -1.70 pp |
| **Lung** | 81.31% | 81.74% | +0.43 pp | 81.12% | -0.19 pp | 82.08% | +0.77 pp |
| **FSS1000** | 69.85% | 65.73% | -4.12 pp | 69.89% | +0.04 pp | 66.49% | -3.36 pp |
| **SUIM** | 39.56% | 35.75% | -3.81 pp | 39.35% | -0.21 pp | 35.88% | -3.68 pp |
| **AVERAGE** | **56.37%** | **54.53%** | **-1.84 pp** | **56.15%** | **-0.22 pp** | **54.62%** | **-1.75 pp** |

---

## 5. Comparison with Published Paper Reference Values

> [!NOTE]
> Paper reference values are reported from the original CVPR 2024 publication (evaluated over 1,000 random episodes on GPU cluster).
> They serve as an empirical reference point for domain difficulty, and are not treated as direct comparison targets due to differing protocols (1,000 random episodes in the original paper vs. 100 deterministic manifest-controlled episodes under seed=42 here).

| Dataset | Published Paper Reference | E0 (Base) | E1 (Adp) | E2 (Fus) | E3 (Prop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 42.3% | 44.74% | 44.08% | 44.44% | 43.73% |
| **ISIC** | 41.8% | 43.29% | 41.80% | 42.68% | 41.72% |
| **Lung** | 80.0% | 81.32% | 81.90% | 81.16% | 82.21% |
| **FSS1000** | 69.3% | 69.85% | 65.73% | 69.89% | 66.49% |
| **SUIM** | 35.0% | 37.01% | 35.16% | 36.68% | 35.35% |

---

## 6. Scientific Analysis (Phân Tích Khoa Học & Giải Đáp 4 Câu Hỏi Trọng Tâm)

### Q1: Adapter cải tiến (Depthwise Separable 3×3) có thực sự hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-1.51 pp** (Mean Episode-IoU: **-1.84 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *DeepGlobe*: Cumulative mIoU biến thiên **-0.66 pp** (44.74% -> 44.08%), Mean Episode-IoU biến thiên **-0.21 pp** (44.06% -> 43.85%).
  - *ISIC*: Cumulative mIoU biến thiên **-1.49 pp** (43.29% -> 41.80%), Mean Episode-IoU biến thiên **-1.50 pp** (47.07% -> 45.57%).
  - *Lung*: Cumulative mIoU biến thiên **+0.58 pp** (81.32% -> 81.90%), Mean Episode-IoU biến thiên **+0.43 pp** (81.31% -> 81.74%).
  - *FSS1000*: Cumulative mIoU biến thiên **-4.12 pp** (69.85% -> 65.73%), Mean Episode-IoU biến thiên **-4.12 pp** (69.85% -> 65.73%).
  - *SUIM*: Cumulative mIoU biến thiên **-1.85 pp** (37.01% -> 35.16%), Mean Episode-IoU biến thiên **-3.81 pp** (39.56% -> 35.75%).
- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Inductive bias mở rộng receptive field từ 1×1 sang 3×3 không mang lại cải thiện đồng đều trên mọi miền dữ liệu. Kernel 3×3 mang lại cải thiện nhẹ trên miền Lung (+0.58 pp), nhưng cho hiệu năng thấp hơn trên các miền đa lớp phức tạp hoặc độ tương phản thấp (như FSS-1000 -4.12 pp, SUIM -1.85 pp, ISIC -1.49 pp) dưới điều kiện 1-shot SGD trực tuyến.

### Q2: Fusion cải tiến (Softmax Margin Fusion) có thực sự hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-0.27 pp** (Mean Episode-IoU: **-0.22 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *DeepGlobe*: Cumulative mIoU biến thiên **-0.30 pp** (44.74% -> 44.44%), Mean Episode-IoU biến thiên **-0.29 pp** (44.06% -> 43.77%).
  - *ISIC*: Cumulative mIoU biến thiên **-0.61 pp** (43.29% -> 42.68%), Mean Episode-IoU biến thiên **-0.46 pp** (47.07% -> 46.61%).
  - *Lung*: Cumulative mIoU biến thiên **-0.16 pp** (81.32% -> 81.16%), Mean Episode-IoU biến thiên **-0.19 pp** (81.31% -> 81.12%).
  - *FSS1000*: Cumulative mIoU biến thiên **+0.04 pp** (69.85% -> 69.89%), Mean Episode-IoU biến thiên **+0.04 pp** (69.85% -> 69.89%).
  - *SUIM*: Cumulative mIoU biến thiên **-0.33 pp** (37.01% -> 36.68%), Mean Episode-IoU biến thiên **-0.21 pp** (39.56% -> 39.35%).
- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Softmax Margin Fusion điều chỉnh trọng số tầng dựa trên khoảng cách prototype giữa foreground và background. Trên FSS-1000, cơ chế này nhích nhẹ (+0.04 pp), trên các miền còn lại hiệu năng tương đối ổn định và bám sát baseline E0 (dao động trong khoảng -0.16 pp đến -0.61 pp).

### Q3: Phương pháp đề xuất kết hợp (E3: DW3×3 + Softmax Margin) có hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-1.34 pp** (Mean Episode-IoU: **-1.75 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *DeepGlobe*: Cumulative mIoU đạt **43.73%** (-1.01 pp so với E0), Mean Episode-IoU đạt **43.30%** (-0.76 pp so với E0).
  - *ISIC*: Cumulative mIoU đạt **41.72%** (-1.57 pp so với E0), Mean Episode-IoU đạt **45.37%** (-1.70 pp so với E0).
  - *Lung*: Cumulative mIoU đạt **82.21%** (+0.89 pp so với E0), Mean Episode-IoU đạt **82.08%** (+0.77 pp so với E0).
  - *FSS1000*: Cumulative mIoU đạt **66.49%** (-3.36 pp so với E0), Mean Episode-IoU đạt **66.49%** (-3.36 pp so với E0).
  - *SUIM*: Cumulative mIoU đạt **35.35%** (-1.66 pp so với E0), Mean Episode-IoU đạt **35.88%** (-3.68 pp so với E0).
- **Kết luận Q3**: Cấu hình kết hợp E3 cải thiện kết quả so với baseline E0 trên miền Lung (+0.89 pp Cumulative mIoU, +0.77 pp Mean Episode-IoU, đạt 82.21%). Tuy nhiên, không thể kết luận đơn giản là E3 'vượt paper' ở miền này (82.21% vs 80.0%) vì hai bên sử dụng protocol khác biệt: bài báo gốc đánh giá trên 1.000 episodes lấy mẫu ngẫu nhiên, trong khi thực nghiệm ở đây đánh giá trên 100 episodes cố định theo manifest (seed=42). Xét trên quy mô trung bình 5 benchmark, hiệu năng của E3 thấp hơn E0 (-1.34 pp Cumulative mIoU, -1.75 pp Mean Episode-IoU), khẳng định tính chất phụ thuộc miền (domain-dependent) của inductive bias kết hợp.

### Q4: Có tương tác (interaction) giữa Adapter và Fusion không?
- **Định lượng tương tác**: Giá trị $\Delta_{Combined} - (\Delta_{Adapter} + \Delta_{Fusion})$ trung bình là **+0.44 pp** (Cumulative mIoU).
  - *DeepGlobe*: Tương tác = **-0.05 pp** (Combined: -1.01 pp vs Tổng tuyến tính: -0.96 pp).
  - *ISIC*: Tương tác = **+0.53 pp** (Combined: -1.57 pp vs Tổng tuyến tính: -2.10 pp).
  - *Lung*: Tương tác = **+0.47 pp** (Combined: +0.89 pp vs Tổng tuyến tính: +0.42 pp).
  - *FSS1000*: Tương tác = **+0.72 pp** (Combined: -3.36 pp vs Tổng tuyến tính: -4.08 pp).
  - *SUIM*: Tương tác = **+0.52 pp** (Combined: -1.66 pp vs Tổng tuyến tính: -2.18 pp).
- **Kết luận Q4**: Tương tác phi tuyến tính có biểu hiện rõ rệt, đặc biệt là super-additive trên Lung (+0.47 pp) và FSS-1000 (+0.72 pp), chứng minh việc kết hợp hai module không đơn thuần là phép cộng tuyến tính độc lập.

---

## 7. Protocol Transparency & Limitations

- **Protocol Control**: Toàn bộ 20 thử nghiệm thực thi cố định với `seed=42`, `nshot=1`, 25 epochs online SGD per episode trên GPU Tesla T4 (Modal Cloud). 100% 5 dataset đều sử dụng explicit deterministic manifests với hash SHA256 đã kiểm chứng.
- **Zero Tuning**: Các siêu tham số (learning rate 0.01, l0=3, temperature 1.0, threshold max_otsu_mean) được đóng băng tuyệt đối xuyên suốt 20 runs.
- **Limitations**: 
  1. Quy mô đánh giá đạt 100 episodes chuẩn mực cho mỗi dataset (tổng 2.000 episodes toàn suite), mang lại ước lượng ổn định hơn (more stable estimate) so với các thử nghiệm quy mô nhỏ trước đây.
  2. Thực nghiệm thực hiện trên 1 seed chuẩn hóa (seed=42), các nghiên cứu tương lai có thể mở rộng lên multi-seed (ví dụ: seeds 42, 123, 999) để đo đạc khoảng tin cậy (confidence intervals).
  3. Toàn bộ mã nguồn, trọng số và episode manifests được công khai minh bạch tại kho lưu trữ KTD1108/ABCDFSS.
