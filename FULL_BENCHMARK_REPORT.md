# FULL BENCHMARK REPORT: E0–E3 EVALUATION ACROSS 5 DATASETS

## Đề Tài: Cross-Domain Few-Shot Semantic Segmentation (CD-FSS)
### Khảo Sát Độc Lập Ablation Study: Original ABCDFSS Baseline vs. Proposed Architecture

---

## 1. Experimental Protocol (Quy Trình Thực Nghiệm Chuẩn Hóa)

Toàn bộ 20 thử nghiệm (5 datasets × 4 cấu hình) được thực thi nghiêm ngặt theo đúng protocol cố định:
Each configuration was evaluated on 20 seed-controlled episodes per dataset using seed=42. The Lung benchmark additionally used an explicit episode manifest to ensure a fixed episode set.

```yaml
seed: 42
nshot: 1
image_size: 400x400
adapt_to: every-episode (Algorithm 2: Test-Time Online SGD, 25 epochs)
learning_rate: 0.01
out_channels: 64
l0: 3 (Intermediate resolution 50x50)
threshold: max(Otsu, mean) with drop_least=0.05
episodes_per_run: 20 seed-controlled episodes (Lung: explicit manifest; others: seed-controlled runtime sampling)
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
> For single-class target domains (Lung, ISIC), this corresponds to the aggregated foreground ratio $\frac{\sum |P_i \cap G_i|}{\sum |P_i \cup G_i|}$. For multi-class benchmarks (DeepGlobe, FSS-1000, SUIM), intersection and union are aggregated per semantic class $c$ before computing the macro-average.

| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adapter | E2 (Fus) | $\Delta$ Fusion | E3 (Prop) | $\Delta$ Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 50.83% | 50.83% | +0.00 pp | 50.76% | -0.07 pp | 50.29% | -0.54 pp |
| **ISIC** | 41.17% | 39.71% | -1.46 pp | 40.96% | -0.21 pp | 40.52% | -0.65 pp |
| **Lung** | 78.69% | 78.84% | +0.15 pp | 79.32% | +0.63 pp | 80.09% | +1.40 pp |
| **FSS1000** | 80.08% | 79.95% | -0.13 pp | 80.19% | +0.11 pp | 79.60% | -0.48 pp |
| **SUIM** | 38.24% | 35.03% | -3.21 pp | 37.67% | -0.57 pp | 34.75% | -3.49 pp |
| **AVERAGE** | **57.80%** | **56.87%** | **-0.93 pp** | **57.78%** | **-0.02 pp** | **57.05%** | **-0.75 pp** |

---

## 4. Benchmark Results — Mean Episode-IoU (%)

Mean Episode-IoU denotes the unweighted average of individual episode IoU values:
$$\text{Mean Episode-IoU} = \frac{1}{N} \sum_{i=1}^N \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$

*(Reference Foreground-Background IoU metric)*:
$$FB\text{-IoU} = \frac{1}{2} \left( \frac{\sum_{i=1}^N |P_{fg, i} \cap G_{fg, i}|}{\sum_{i=1}^N |P_{fg, i} \cup G_{fg, i}|} + \frac{\sum_{i=1}^N |P_{bg, i} \cap G_{bg, i}|}{\sum_{i=1}^N |P_{bg, i} \cup G_{bg, i}|} \right)$$

| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adapter | E2 (Fus) | $\Delta$ Fusion | E3 (Prop) | $\Delta$ Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 51.68% | 52.08% | +0.40 pp | 51.44% | -0.24 pp | 51.51% | -0.17 pp |
| **ISIC** | 48.50% | 46.99% | -1.51 pp | 48.24% | -0.26 pp | 47.78% | -0.72 pp |
| **Lung** | 78.34% | 78.23% | -0.11 pp | 78.95% | +0.61 pp | 79.41% | +1.07 pp |
| **FSS1000** | 80.21% | 80.10% | -0.11 pp | 80.35% | +0.14 pp | 79.73% | -0.48 pp |
| **SUIM** | 38.65% | 33.11% | -5.54 pp | 38.35% | -0.30 pp | 32.97% | -5.68 pp |
| **AVERAGE** | **59.48%** | **58.10%** | **-1.37 pp** | **59.47%** | **-0.01 pp** | **58.28%** | **-1.20 pp** |

---

## 5. Comparison with Published Paper Reference Values

> [!NOTE]
> Paper reference values are reported from the original CVPR 2024 publication (evaluated over 1,000 episodes on GPU cluster).
> They serve as an empirical reference point for domain difficulty, and are not treated as exact reproduction targets under the 20-episode seed-controlled evaluation protocol.

| Dataset | Published Paper Reference | E0 (Base) | E1 (Adp) | E2 (Fus) | E3 (Prop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 42.3% | 50.83% | 50.83% | 50.76% | 50.29% |
| **ISIC** | 41.8% | 41.17% | 39.71% | 40.96% | 40.52% |
| **Lung** | 80.0% | 78.69% | 78.84% | 79.32% | 80.09% |
| **FSS1000** | 69.3% | 80.08% | 79.95% | 80.19% | 79.60% |
| **SUIM** | 35.0% | 38.24% | 35.03% | 37.67% | 34.75% |

---

## 6. Scientific Analysis (Phân Tích Khoa Học & Giải Đáp 4 Câu Hỏi Trọng Tâm)

### Q1: Adapter cải tiến (Depthwise Separable 3×3) có thực sự hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-0.93 pp** (Mean Episode-IoU: **-1.37 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *Lung*: Cumulative mIoU tăng nhẹ **+0.15 pp** (78.69% -> 78.84%), trong khi Mean Episode-IoU biến thiên -0.11 pp (78.34% -> 78.23%).
  - *DeepGlobe*: Cumulative mIoU giữ nguyên (+0.00 pp, 50.83%), trong khi Mean Episode-IoU tăng **+0.40 pp** (51.68% -> 52.08%).
  - *FSS-1000*: Cumulative mIoU giảm nhẹ -0.13 pp (80.08% -> 79.95%), Mean Episode-IoU giảm -0.11 pp (80.21% -> 80.10%).
  - *ISIC*: DW3×3 cho kết quả thấp hơn baseline: Cumulative mIoU giảm **-1.46 pp** (41.17% -> 39.71%), Mean Episode-IoU giảm **-1.51 pp** (48.50% -> 46.99%).
  - *SUIM*: DW3×3 cho kết quả thấp hơn baseline: Cumulative mIoU giảm **-3.21 pp** (38.24% -> 35.03%), Mean Episode-IoU giảm **-5.54 pp** (38.65% -> 33.11%).
- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Inductive bias mở rộng receptive field từ 1×1 sang 3×3 không mang lại cải thiện đồng đều trên mọi miền dữ liệu. Một giả thuyết khả dĩ là kernel 3×3 với receptive field lớn hơn có thể hữu ích ở các miền có cấu trúc biên rõ (như ảnh giải phẫu hoặc đường sá), nhưng kém phù hợp hơn trên các miền có biên độ tương phản thấp hoặc nhiễu tán xạ cao (như ISIC và SUIM) dưới điều kiện 1-shot SGD trực tuyến. Tuy nhiên, giả thuyết này cần thêm các thực nghiệm kiểm chứng có kiểm soát.

### Q2: Fusion cải tiến (Softmax Margin Fusion) có thực sự hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-0.02 pp** (Mean Episode-IoU: **-0.01 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *Lung*: Ghi nhận mức cải thiện rõ nét nhất: Cumulative mIoU tăng **+0.63 pp** (78.69% -> 79.32%), Mean Episode-IoU tăng **+0.61 pp** (78.34% -> 78.95%).
  - *FSS-1000*: Cumulative mIoU tăng nhẹ **+0.11 pp** (80.08% -> 80.19%), Mean Episode-IoU tăng **+0.14 pp** (80.35% vs 80.21%).
  - *DeepGlobe*: Cumulative mIoU biến thiên -0.07 pp (50.83% -> 50.76%), Mean Episode-IoU biến thiên -0.24 pp (51.68% -> 51.44%).
  - *ISIC*: Cumulative mIoU biến thiên -0.21 pp (41.17% -> 40.96%), Mean Episode-IoU biến thiên -0.26 pp (48.50% -> 48.24%).
  - *SUIM*: Cumulative mIoU biến thiên -0.57 pp (38.24% -> 37.67%), Mean Episode-IoU biến thiên -0.30 pp (38.65% -> 38.35%).
- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Softmax Margin Fusion điều chỉnh trọng số tầng dựa trên khoảng cách prototype giữa foreground và background. Trên miền Lung, cơ chế này giúp tăng tỷ trọng của các tầng có độ phân tách hình học cao. Trên 4 miền còn lại, kết quả dao động sát mức baseline (biến thiên trung bình toàn benchmark là -0.02 pp).

### Q3: Phương pháp đề xuất kết hợp (E3: DW3×3 + Softmax Margin) có hiệu quả không?
- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **-0.75 pp** (Mean Episode-IoU: **-1.20 pp**).
- **Chi tiết theo từng miền dữ liệu**:
  - *Lung*: Cấu hình kết hợp E3 đạt **80.09% Cumulative mIoU** (+1.40 pp so với E0 78.69%) và **79.41% Mean Episode-IoU** (+1.07 pp so với E0 78.34%). Con số 80.09% nằm sát mốc tham chiếu 80.0% được công bố trong bài báo gốc.
  - *DeepGlobe*: Cumulative mIoU đạt 50.29% (-0.54 pp so với E0), Mean Episode-IoU đạt 51.51% (-0.17 pp so với E0).
  - *FSS-1000*: Cumulative mIoU đạt 79.60% (-0.48 pp so với E0), Mean Episode-IoU đạt 79.73% (-0.48 pp so với E0).
  - *ISIC*: Cumulative mIoU đạt 40.52% (-0.65 pp so với E0), Mean Episode-IoU đạt 47.78% (-0.72 pp so với E0).
  - *SUIM*: Cumulative mIoU đạt 34.75% (-3.49 pp so với E0), Mean Episode-IoU đạt 32.97% (-5.68 pp so với E0).
- **Kết luận Q3**: Cấu hình kết hợp E3 cải thiện kết quả rõ ràng trên miền Lung (+1.40 pp Cumulative mIoU), nhưng không đem lại cải thiện đồng đều trên toàn bộ 5 benchmark. Hiệu năng trung bình của E3 trên 5 dataset thấp hơn E0 (-0.75 pp Cumulative mIoU, -1.20 pp Mean Episode-IoU), cho thấy hiệu quả của phương pháp kết hợp mang tính phụ thuộc miền (domain-dependent) thay vì ưu việt phổ quát.

### Q4: Có tương tác (interaction) giữa Adapter và Fusion không?
- **Định lượng tương tác**: Giá trị $\Delta_{Combined} - (\Delta_{Adapter} + \Delta_{Fusion})$ trung bình là **+0.20 pp** (Cumulative mIoU).
  - *Trên Lung*: $\Delta_{Combined} (+1.40\text{ pp}) > \Delta_{Adapter} (+0.15\text{ pp}) + \Delta_{Fusion} (+0.63\text{ pp}) = +0.78\text{ pp}$. Tương tác quan sát được là **+0.62 pp**, cho thấy kết quả kết hợp trên tập đánh giá này có dạng super-additive.
  - *Trên các miền còn lại*: Giá trị tương tác dao động: DeepGlobe (-0.47 pp), ISIC (+1.02 pp), FSS-1000 (-0.46 pp), SUIM (+0.29 pp).
- **Kết luận Q4**: Các kết quả quan sát cho thấy có sự tương tác giữa hai thành phần trên từng tập dữ liệu cụ thể (đặc biệt là Lung), nhưng để khẳng định hiệu ứng cộng hưởng (synergy) có ý nghĩa thống kê tổng quát thì cần mở rộng thêm các thực nghiệm đa seed.

---

## 7. Protocol Transparency & Limitations

- **Protocol Control**: Toàn bộ thử nghiệm thực thi cố định với `seed=42`, `nshot=1`, 25 epochs online SGD per episode. Dataset Lung sử dụng manifest 20 episode cố định; 4 dataset còn lại sử dụng runtime sampling có kiểm soát seed.
- **Zero Tuning**: Các siêu tham số (learning rate 0.01, l0=3, temperature 1.0, threshold max_otsu_mean) được giữ nguyên hoàn toàn xuyên suốt 20 runs.
- **Limitations**: 
  1. Quy mô đánh giá gồm 20 episode cho mỗi dataset (do hạn chế tính toán trên CPU), không thay thế cho đánh giá 1.000 episode quy mô lớn trên GPU cluster.
  2. Thực nghiệm thực hiện trên 1 seed duy nhất (seed=42), chưa đủ để thực hiện kiểm định ý nghĩa thống kê (t-test / ANOVA).
  3. Các nhận định về nguyên nhân vật lý/hình ảnh (ví dụ: tán xạ dưới nước, sắc tố da) hiện dừng ở mức giả thuyết khoa học hợp lý, cần thêm kiểm chứng phân rã lỗi (error visual breakdown).
