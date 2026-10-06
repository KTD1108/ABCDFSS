# Full Benchmark Report Audit

## Audit Date
2026-10-06

## Repository Commit
`1553a4d` (Branch: `main`)

## Benchmark Commit
`0147b5a` (`feat(benchmark): complete full CD-FSS benchmark across 5 datasets (E0-E3) and add comprehensive report`)

---

## 1. Artifact Verification
Toàn bộ 20/20 raw artifact files (`run_result.json`, `summary_records.jsonl`, và file `*.log`) được kiểm tra trực tiếp từ `results/full_benchmark/`:

- **DeepGlobe** (4 runs: E0, E1, E2, E3): **PASS** (Tất cả 4 runs đủ 20 episodes, raw IoUs đủ 20 giá trị, seed=42)
- **ISIC** (4 runs: E0, E1, E2, E3): **PASS** (Tất cả 4 runs đủ 20 episodes, raw IoUs đủ 20 giá trị, seed=42)
- **Lung** (4 runs: E0, E1, E2, E3): **PASS** (Tất cả 4 runs đủ 20 episodes, manifest cố định `lung_seed42_20episodes.json`)
- **FSS1000** (4 runs: E0, E1, E2, E3): **PASS** (Tất cả 4 runs đủ 20 episodes, raw IoUs đủ 20 giá trị, seed=42)
- **SUIM** (4 runs: E0, E1, E2, E3): **PASS** (Tất cả 4 runs đủ 20 episodes, raw IoUs đủ 20 giá trị, seed=42)

**Tổng kết Artifact**: **20/20 runs PASS**. Không có artifact nào bị thiếu, hỏng hay phải ước lượng/tạo giả.

---

## 2. Experiment Verification
Đối chiếu mapping cấu hình kiến trúc thực tế giữa code `src/` và artifacts:

- **E0 (Original ABCDFSS Baseline)**: **PASS**
  - Adapter: `conv1x1` (Pointwise Conv $1\times 1$, in_c $\to$ 64)
  - Layer Fusion: `mean` (Flat average across layers $l \ge 3$)
- **E1 (Adapter Ablation)**: **PASS**
  - Adapter: `depthwise_separable_3x3` (DW Conv $3\times 3$ groups=in_c + PW Conv $1\times 1$)
  - Layer Fusion: `mean`
- **E2 (Fusion Ablation)**: **PASS**
  - Adapter: `conv1x1`
  - Layer Fusion: `softmax_margin` (Prototype cosine distance margin weighting, temp=1.0)
- **E3 (Full Proposed Method)**: **PASS**
  - Adapter: `depthwise_separable_3x3`
  - Layer Fusion: `softmax_margin`

---

## 3. Protocol Verification
Kiểm tra protocol thực thi thực tế trong `config` của từng `run_result.json` và log:

- `seed`: 42 (**PASS**)
- `nshot`: 1 (**PASS**)
- `image_size`: 400x400 (**PASS**)
- `adapt_to`: `every-episode` (Algorithm 2: Test-Time Online SGD, 25 epochs per episode, lr=0.01) (**PASS**)
- `l0`: 3 (Intermediate resolution 50x50) (**PASS**)
- `out_channels`: 64 (**PASS**)
- `threshold_method`: `pred_mean` (`max(Otsu, mean)` with `drop_least=0.05`) (**PASS**)
- `backbone`: ResNet-50 (Pre-ReLU unclipped features, ImageNet weights frozen) (**PASS**)
- **Phân định tập episode**: 
  - Đã audit và đính chính mô tả protocol: Chỉ có benchmark **Lung** sử dụng explicit manifest (`lung_seed42_20episodes.json`).
  - 4 benchmark còn lại (**DeepGlobe, ISIC, FSS-1000, SUIM**) sử dụng seed-controlled runtime sampling (`seed=42`, `bsz=1`, `nworker=0`). Không claim "20 fixed episodes were used for every dataset" sai thực tế.

---

## 4. Metric Verification
Đọc trực tiếp mã nguồn `src/metrics/metrics.py` (lines 43–81):

1. **Cumulative mIoU**:
   - Implementation thực tế là **class-wise aggregated IoU** rồi macro-average qua tất cả các class được đánh giá:
     $$\text{IoU}_c = \frac{\sum_{i \in \mathcal{E}_c} |P_i \cap G_i|}{\sum_{i \in \mathcal{E}_c} |P_i \cup G_i|}, \quad \text{Cumulative mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c$$
   - Đối với bài toán đơn lớp mục tiêu (Lung, ISIC: $C=\{0\}$), công thức này tương đương với tỷ số foreground toàn cục $\frac{\sum |P_i \cap G_i|}{\sum |P_i \cup G_i|}$.
   - Đối với bài toán đa lớp (DeepGlobe, FSS-1000, SUIM), phép tính tích lũy giao/hợp theo từng semantic class $c$ trước khi lấy trung bình macro.
   - **Đã sửa lỗi**: Báo cáo trước đó ghi công thức tổng quát rút gọn sai dạng đa lớp; đã cập nhật định nghĩa chính xác theo code.

2. **Mean Episode-IoU**:
   - Implementation thực tế: Trung bình số học không trọng số của IoU từng episode:
     $$\text{Mean Episode-IoU} = \frac{1}{N} \sum_{i=1}^N \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$
   - Độc lập hoàn toàn với Cumulative mIoU và phản ánh độ ổn định giữa các episode.

3. **FB-IoU**:
   - Implementation thực tế: Trung bình của Foreground IoU và Background IoU:
     $$FB\text{-IoU} = \frac{1}{2} \left( \frac{\sum_{i=1}^N |P_{fg, i} \cap G_{fg, i}|}{\sum_{i=1}^N |P_{fg, i} \cup G_{fg, i}|} + \frac{\sum_{i=1}^N |P_{bg, i} \cap G_{bg, i}|}{\sum_{i=1}^N |P_{bg, i} \cup G_{bg, i}|} \right)$$

---

## 5. Numerical Consistency
Đối chiếu 100% giữa raw artifacts `run_result.json` và bảng báo cáo `FULL_BENCHMARK_REPORT.md`:

### A. Cumulative mIoU (%)
| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adp | E2 (Fus) | $\Delta$ Fus | E3 (Prop) | $\Delta$ Comb | Interaction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 50.83% | 50.83% | +0.00 pp | 50.76% | -0.07 pp | 50.29% | -0.54 pp | -0.47 pp |
| **ISIC** | 41.17% | 39.71% | -1.46 pp | 40.96% | -0.21 pp | 40.52% | -0.65 pp | +1.02 pp |
| **Lung** | 78.69% | 78.84% | +0.15 pp | 79.32% | +0.63 pp | 80.09% | +1.40 pp | +0.62 pp |
| **FSS1000** | 80.08% | 79.95% | -0.13 pp | 80.19% | +0.11 pp | 79.60% | -0.48 pp | -0.46 pp |
| **SUIM** | 38.24% | 35.03% | -3.21 pp | 37.67% | -0.57 pp | 34.75% | -3.49 pp | +0.29 pp |
| **AVERAGE** | **57.80%** | **56.87%** | **-0.93 pp** | **57.78%** | **-0.02 pp** | **57.05%** | **-0.75 pp** | **+0.20 pp** |

### B. Mean Episode-IoU (%)
| Dataset | E0 (Base) | E1 (Adp) | $\Delta$ Adp | E2 (Fus) | $\Delta$ Fus | E3 (Prop) | $\Delta$ Comb | Interaction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepGlobe** | 51.68% | 52.08% | +0.40 pp | 51.44% | -0.24 pp | 51.51% | -0.17 pp | -0.33 pp |
| **ISIC** | 48.50% | 46.99% | -1.51 pp | 48.24% | -0.26 pp | 47.78% | -0.72 pp | +1.05 pp |
| **Lung** | 78.34% | 78.23% | -0.11 pp | 78.95% | +0.61 pp | 79.41% | +1.07 pp | +0.57 pp |
| **FSS1000** | 80.21% | 80.10% | -0.11 pp | 80.35% | +0.14 pp | 79.73% | -0.48 pp | -0.51 pp |
| **SUIM** | 38.65% | 33.11% | -5.54 pp | 38.35% | -0.30 pp | 32.97% | -5.68 pp | +0.16 pp |
| **AVERAGE** | **59.48%** | **58.10%** | **-1.37 pp** | **59.47%** | **-0.01 pp** | **58.28%** | **-1.20 pp** | **+0.19 pp** |

**Kết luận số học**: 100% số liệu trong bảng, các hiệu số ablation $\Delta$ và tương tác đều khớp chính xác với raw artifacts (`run_result.json`).

---

## 6. Scientific Claim Audit
Đã rà soát và loại bỏ/điều chỉnh các tuyên bố vượt quá phạm vi bằng chứng thực nghiệm:

1. **Về tính ưu việt tổng quát (Universal Superiority)**:
   - Đã loại bỏ nhận định cho rằng Adapter DW3×3 hay E3 "cải thiện vượt trội".
   - Báo cáo đã ghi nhận trung thực: Trên mức trung bình 5 benchmark, E3 (57.05% Cumulative, 58.28% Episode) có kết quả thấp hơn baseline E0 (57.80% Cumulative, 59.48% Episode).
   - Lợi ích của E3 có tính chất phụ thuộc miền (domain-dependent), chỉ cải thiện rõ rệt trên Lung (+1.40 pp Cumulative, +1.07 pp Episode).
2. **Về suy diễn nhân quả chưa chứng minh (Unproven Causal Inferences)**:
   - Đã tách bạch rõ giữa **Quan sát (Observation)**, **Diễn giải (Interpretation)**, và **Giả thuyết (Hypothesis)**.
   - Không khẳng định kernel 3×3 "overfit vào sắc tố/nhiễu tán xạ", mà mô tả chính xác rằng kernel 3×3 cho kết quả thấp hơn baseline trên ISIC và SUIM dưới protocol hiện tại, và xếp nhận định về receptive field vào dạng giả thuyết khoa học cần thêm kiểm chứng phân rã lỗi.
3. **Về so sánh với bài báo công bố (Paper Comparison)**:
   - Đã loại bỏ khẳng định "tái lập chính xác/exact reproduction bài báo".
   - Ghi rõ điểm số Lung E3 (80.09%) là "nằm sát mốc giá trị tham chiếu 80.0% được công bố trong bài báo gốc", và làm rõ điều kiện khác biệt về quy mô (20 episodes trên CPU vs 1.000 episodes trên GPU cluster).
4. **Về hiệu ứng cộng hưởng (Synergy)**:
   - Không tuyên bố "chứng minh hiệu ứng cộng hưởng", mà chỉ ghi nhận giá trị tương tác dương dạng super-additive (+0.62 pp) quan sát được trên tập dữ liệu Lung cụ thể, đồng thời lưu ý cần thực nghiệm đa seed để đánh giá ý nghĩa thống kê.

---

## 7. Changes Made

1. **`FULL_BENCHMARK_REPORT.md`**:
   - Cập nhật định nghĩa toán học và giải thích chi tiết của `Cumulative mIoU` (class-wise aggregated macro-average) và `Mean Episode-IoU`.
   - Bổ sung định nghĩa `FB-IoU`.
   - Chuẩn hóa mô tả protocol: làm rõ chỉ có Lung dùng explicit manifest, 4 dataset còn lại dùng seed-controlled runtime sampling.
   - Viết lại toàn bộ Mục 6 (Scientific Analysis) để đảm bảo ngôn ngữ học thuật trung tính, phản ánh đa chiều cả kết quả tích cực lẫn suy giảm, phân tách rõ Observation vs Hypothesis.
   - Cập nhật Mục 7 (Transparency & Limitations) ghi rõ các giới hạn về số lượng episode và số seed.
2. **`generate_full_benchmark_report.py`**:
   - Đồng bộ toàn bộ logic sinh nội dung báo cáo để bất kỳ lần chạy lại nào của script cũng sinh ra văn bản khoa học chuẩn mực đã qua audit.

---

## 8. Remaining Limitations

1. **Kích thước mẫu đánh giá**: Đánh giá 20 episode/dataset là kích thước mẫu có kiểm soát để chạy trên CPU, không thay thế cho đánh giá 1.000 episode quy mô lớn trên cluster GPU.
2. **Số lượng Seed**: Toàn bộ đánh giá thực hiện trên `seed=42`, chưa có độ lệch chuẩn qua nhiều seed để làm kiểm định thống kê ANOVA hay t-test.
3. **Phân rã trực quan (Visual Error Analysis)**: Các giả thuyết về ảnh hưởng của receptive field trên biên mềm/nhiễu chưa có phân tích định tính từng pixel tương ứng.

---

## 9. Final Verdict

### **PASS WITH DOCUMENTATION FIXES**

- Toàn bộ 20/20 benchmark artifacts hợp lệ, hoàn chỉnh và có thể tái lập hoàn toàn từ mã nguồn và dữ liệu gốc.
- Không có bất kỳ thay đổi nào đối với thuật toán hay số liệu thực nghiệm.
- Báo cáo đã được audit và đính chính các định nghĩa metric, mô tả protocol và văn phong khoa học, loại bỏ hoàn toàn các overclaim.
- Tài liệu hiện tại đạt chuẩn khoa học nghiêm ngặt và sẵn sàng được sử dụng làm phần Experimental Results trong báo cáo nghiên cứu.
