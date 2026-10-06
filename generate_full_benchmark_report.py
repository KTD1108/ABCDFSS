#!/usr/bin/env python3
"""
Generate comprehensive FULL_BENCHMARK_REPORT.md and summary tables
from results/full_benchmark/{Dataset}/{Experiment}/run_result.json.
"""

import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "full_benchmark")

DATASETS = ['DeepGlobe', 'ISIC', 'Lung', 'FSS1000', 'SUIM']
EXPERIMENTS = ['E0', 'E1', 'E2', 'E3']

PAPER_REFERENCE = {
    'DeepGlobe': 42.3,
    'ISIC': 41.8,
    'Lung': 80.0,
    'FSS1000': 69.3,
    'SUIM': 35.0
}

def load_all_results():
    matrix = {}
    for ds in DATASETS:
        matrix[ds] = {}
        for exp in EXPERIMENTS:
            res_file = os.path.join(RESULTS_DIR, ds, exp, "run_result.json")
            if os.path.exists(res_file):
                try:
                    with open(res_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    matrix[ds][exp] = {
                        'Cumulative_mIoU': data['metric']['Cumulative_mIoU'],
                        'Mean_Episode_IoU': data['metric']['Mean_Episode_IoU'],
                        'FB-IoU': data['metric']['FB-IoU'],
                        'episodes': data.get('episode_count', 0),
                        'log': data.get('log', '')
                    }
                except Exception as e:
                    matrix[ds][exp] = None
            else:
                matrix[ds][exp] = None
    return matrix

def generate_report():
    matrix = load_all_results()

    lines = []
    lines.append("# FULL BENCHMARK REPORT: E0–E3 EVALUATION ACROSS 5 DATASETS\n")
    lines.append("## Đề Tài: Cross-Domain Few-Shot Semantic Segmentation (CD-FSS)")
    lines.append("### Khảo Sát Độc Lập Ablation Study: Original ABCDFSS Baseline vs. Proposed Architecture\n")
    lines.append("---\n")

    # 1. Protocol & Execution Environment
    lines.append("## 1. Experimental Protocol (Quy Trình Thực Nghiệm Chuẩn Hóa)\n")
    lines.append("Toàn bộ 20 thử nghiệm (5 datasets × 4 cấu hình) được thực thi nghiêm ngặt theo đúng protocol cố định:")
    lines.append("Each configuration was evaluated on 20 seed-controlled episodes per dataset using seed=42. The Lung benchmark additionally used an explicit episode manifest to ensure a fixed episode set.\n")
    lines.append("```yaml")
    lines.append("seed: 42")
    lines.append("nshot: 1")
    lines.append("image_size: 400x400")
    lines.append("adapt_to: every-episode (Algorithm 2: Test-Time Online SGD, 25 epochs)")
    lines.append("learning_rate: 0.01")
    lines.append("out_channels: 64")
    lines.append("l0: 3 (Intermediate resolution 50x50)")
    lines.append("threshold: max(Otsu, mean) with drop_least=0.05")
    lines.append("episodes_per_run: 20 seed-controlled episodes (Lung: explicit manifest; others: seed-controlled runtime sampling)")
    lines.append("backbone: ResNet-50 (Pre-ReLU unclipped features, ImageNet weights frozen)")
    lines.append("```\n")

    # 2. Experiment Definition Matrix
    lines.append("## 2. Experiment Matrix Definition\n")
    lines.append("| Experiment | Adapter Architecture | Layer Fusion Mechanism | Role |")
    lines.append("| :---: | :--- | :--- | :--- |")
    lines.append("| **E0** | Conv $1\\times 1$ (Pointwise) | Mean Fusion | Original ABCDFSS Baseline |")
    lines.append("| **E1** | Depthwise Separable Conv $3\\times 3$ | Mean Fusion | Adapter Ablation (Isolating Adapter) |")
    lines.append("| **E2** | Conv $1\\times 1$ (Pointwise) | Softmax Margin Fusion | Fusion Ablation (Isolating Fusion) |")
    lines.append("| **E3** | Depthwise Separable Conv $3\\times 3$ | Softmax Margin Fusion | Full Proposed Method (Combined) |\n")

    # 3. Main Results Table: Cumulative mIoU
    lines.append("## 3. Benchmark Results — Cumulative mIoU (%)\n")
    lines.append("Cumulative mIoU denotes the mean IoU computed after aggregating class-wise intersection and union statistics across the evaluated episodes:")
    lines.append("$$\\text{IoU}_c = \\frac{\\sum_{i \\in \\mathcal{E}_c} |P_i \\cap G_i|}{\\sum_{i \\in \\mathcal{E}_c} |P_i \\cup G_i|}, \\quad \\text{Cumulative mIoU} = \\frac{1}{|C|} \\sum_{c \\in C} \\text{IoU}_c$$\n")
    lines.append("> [!NOTE]")
    lines.append("> For single-class target domains (Lung, ISIC), this corresponds to the aggregated foreground ratio $\\frac{\\sum |P_i \\cap G_i|}{\\sum |P_i \\cup G_i|}$. For multi-class benchmarks (DeepGlobe, FSS-1000, SUIM), intersection and union are aggregated per semantic class $c$ before computing the macro-average.\n")
    lines.append("| Dataset | E0 (Base) | E1 (Adp) | $\\Delta$ Adapter | E2 (Fus) | $\\Delta$ Fusion | E3 (Prop) | $\\Delta$ Combined |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    e0_cums, e1_cums, e2_cums, e3_cums = [], [], [], []
    d_adp_cums, d_fus_cums, d_comb_cums = [], [], []

    for ds in DATASETS:
        e0 = matrix[ds]['E0']
        e1 = matrix[ds]['E1']
        e2 = matrix[ds]['E2']
        e3 = matrix[ds]['E3']

        e0_val = f"{e0['Cumulative_mIoU']:.2f}%" if e0 else "Pending"
        e1_val = f"{e1['Cumulative_mIoU']:.2f}%" if e1 else "Pending"
        e2_val = f"{e2['Cumulative_mIoU']:.2f}%" if e2 else "Pending"
        e3_val = f"{e3['Cumulative_mIoU']:.2f}%" if e3 else "Pending"

        d_adp_str, d_fus_str, d_comb_str = "—", "—", "—"
        if e0 and e1:
            d_adp = e1['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            d_adp_str = f"{d_adp:+.2f} pp"
            d_adp_cums.append(d_adp)
            e0_cums.append(e0['Cumulative_mIoU'])
            e1_cums.append(e1['Cumulative_mIoU'])
        if e0 and e2:
            d_fus = e2['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            d_fus_str = f"{d_fus:+.2f} pp"
            d_fus_cums.append(d_fus)
            e2_cums.append(e2['Cumulative_mIoU'])
        if e0 and e3:
            d_comb = e3['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            d_comb_str = f"{d_comb:+.2f} pp"
            d_comb_cums.append(d_comb)
            e3_cums.append(e3['Cumulative_mIoU'])

        lines.append(f"| **{ds}** | {e0_val} | {e1_val} | {d_adp_str} | {e2_val} | {d_fus_str} | {e3_val} | {d_comb_str} |")

    if len(d_comb_cums) > 0:
        mean_e0 = np.mean(e0_cums)
        mean_e1 = np.mean(e1_cums) if e1_cums else 0
        mean_e2 = np.mean(e2_cums) if e2_cums else 0
        mean_e3 = np.mean(e3_cums) if e3_cums else 0
        mean_d_adp = np.mean(d_adp_cums) if d_adp_cums else 0
        mean_d_fus = np.mean(d_fus_cums) if d_fus_cums else 0
        mean_d_comb = np.mean(d_comb_cums) if d_comb_cums else 0

        lines.append(f"| **AVERAGE** | **{mean_e0:.2f}%** | **{mean_e1:.2f}%** | **{mean_d_adp:+.2f} pp** | **{mean_e2:.2f}%** | **{mean_d_fus:+.2f} pp** | **{mean_e3:.2f}%** | **{mean_d_comb:+.2f} pp** |")

    lines.append("\n---\n")

    # 4. Main Results Table: Mean Episode-IoU
    lines.append("## 4. Benchmark Results — Mean Episode-IoU (%)\n")
    lines.append("Mean Episode-IoU denotes the unweighted average of individual episode IoU values:")
    lines.append("$$\\text{Mean Episode-IoU} = \\frac{1}{N} \\sum_{i=1}^N \\frac{|P_i \\cap G_i|}{|P_i \\cup G_i|}$$\n")
    lines.append("*(Reference Foreground-Background IoU metric)*:")
    lines.append("$$FB\\text{-IoU} = \\frac{1}{2} \\left( \\frac{\\sum_{i=1}^N |P_{fg, i} \\cap G_{fg, i}|}{\\sum_{i=1}^N |P_{fg, i} \\cup G_{fg, i}|} + \\frac{\\sum_{i=1}^N |P_{bg, i} \\cap G_{bg, i}|}{\\sum_{i=1}^N |P_{bg, i} \\cup G_{bg, i}|} \\right)$$\n")
    lines.append("| Dataset | E0 (Base) | E1 (Adp) | $\\Delta$ Adapter | E2 (Fus) | $\\Delta$ Fusion | E3 (Prop) | $\\Delta$ Combined |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    e0_eps, e1_eps, e2_eps, e3_eps = [], [], [], []
    d_adp_eps, d_fus_eps, d_comb_eps = [], [], []

    for ds in DATASETS:
        e0 = matrix[ds]['E0']
        e1 = matrix[ds]['E1']
        e2 = matrix[ds]['E2']
        e3 = matrix[ds]['E3']

        e0_val = f"{e0['Mean_Episode_IoU']:.2f}%" if e0 else "Pending"
        e1_val = f"{e1['Mean_Episode_IoU']:.2f}%" if e1 else "Pending"
        e2_val = f"{e2['Mean_Episode_IoU']:.2f}%" if e2 else "Pending"
        e3_val = f"{e3['Mean_Episode_IoU']:.2f}%" if e3 else "Pending"

        d_adp_str, d_fus_str, d_comb_str = "—", "—", "—"
        if e0 and e1:
            d_adp = e1['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            d_adp_str = f"{d_adp:+.2f} pp"
            d_adp_eps.append(d_adp)
            e0_eps.append(e0['Mean_Episode_IoU'])
            e1_eps.append(e1['Mean_Episode_IoU'])
        if e0 and e2:
            d_fus = e2['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            d_fus_str = f"{d_fus:+.2f} pp"
            d_fus_eps.append(d_fus)
            e2_eps.append(e2['Mean_Episode_IoU'])
        if e0 and e3:
            d_comb = e3['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            d_comb_str = f"{d_comb:+.2f} pp"
            d_comb_eps.append(d_comb)
            e3_eps.append(e3['Mean_Episode_IoU'])

        lines.append(f"| **{ds}** | {e0_val} | {e1_val} | {d_adp_str} | {e2_val} | {d_fus_str} | {e3_val} | {d_comb_str} |")

    if len(d_comb_eps) > 0:
        mean_e0 = np.mean(e0_eps)
        mean_e1 = np.mean(e1_eps) if e1_eps else 0
        mean_e2 = np.mean(e2_eps) if e2_eps else 0
        mean_e3 = np.mean(e3_eps) if e3_eps else 0
        mean_d_adp = np.mean(d_adp_eps) if d_adp_eps else 0
        mean_d_fus = np.mean(d_fus_eps) if d_fus_eps else 0
        mean_d_comb = np.mean(d_comb_eps) if d_comb_eps else 0

        lines.append(f"| **AVERAGE** | **{mean_e0:.2f}%** | **{mean_e1:.2f}%** | **{mean_d_adp:+.2f} pp** | **{mean_e2:.2f}%** | **{mean_d_fus:+.2f} pp** | **{mean_e3:.2f}%** | **{mean_d_comb:+.2f} pp** |")

    lines.append("\n---\n")

    # 5. Reference Comparison with CVPR 2024 Paper
    lines.append("## 5. Comparison with Published Paper Reference Values\n")
    lines.append("> [!NOTE]")
    lines.append("> Paper reference values are reported from the original CVPR 2024 publication (evaluated over 1,000 episodes on GPU cluster).")
    lines.append("> They serve as an empirical reference point for domain difficulty, and are not treated as exact reproduction targets under the 20-episode seed-controlled evaluation protocol.\n")
    lines.append("| Dataset | Published Paper Reference | E0 (Base) | E1 (Adp) | E2 (Fus) | E3 (Prop) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for ds in DATASETS:
        pref = PAPER_REFERENCE[ds]
        e0 = f"{matrix[ds]['E0']['Cumulative_mIoU']:.2f}%" if matrix[ds]['E0'] else "Pending"
        e1 = f"{matrix[ds]['E1']['Cumulative_mIoU']:.2f}%" if matrix[ds]['E1'] else "Pending"
        e2 = f"{matrix[ds]['E2']['Cumulative_mIoU']:.2f}%" if matrix[ds]['E2'] else "Pending"
        e3 = f"{matrix[ds]['E3']['Cumulative_mIoU']:.2f}%" if matrix[ds]['E3'] else "Pending"
        lines.append(f"| **{ds}** | {pref:.1f}% | {e0} | {e1} | {e2} | {e3} |")

    lines.append("\n---\n")

    # 6. Scientific Analysis (Giải Đáp 4 Câu Hỏi Khoa Học)
    lines.append("## 6. Scientific Analysis (Phân Tích Khoa Học & Giải Đáp 4 Câu Hỏi Trọng Tâm)\n")
    
    # Q1
    lines.append("### Q1: Adapter cải tiến (Depthwise Separable 3×3) có thực sự hiệu quả không?")
    avg_adp_cum = np.mean(d_adp_cums) if d_adp_cums else 0
    avg_adp_ep = np.mean(d_adp_eps) if d_adp_eps else 0
    lines.append(f"- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **{avg_adp_cum:+.2f} pp** (Mean Episode-IoU: **{avg_adp_ep:+.2f} pp**).")
    lines.append("- **Chi tiết theo từng miền dữ liệu**:")
    lines.append("  - *Lung*: Cumulative mIoU tăng nhẹ **+0.15 pp** (78.69% -> 78.84%), trong khi Mean Episode-IoU biến thiên -0.11 pp (78.34% -> 78.23%).")
    lines.append("  - *DeepGlobe*: Cumulative mIoU giữ nguyên (+0.00 pp, 50.83%), trong khi Mean Episode-IoU tăng **+0.40 pp** (51.68% -> 52.08%).")
    lines.append("  - *FSS-1000*: Cumulative mIoU giảm nhẹ -0.13 pp (80.08% -> 79.95%), Mean Episode-IoU giảm -0.11 pp (80.21% -> 80.10%).")
    lines.append("  - *ISIC*: DW3×3 cho kết quả thấp hơn baseline: Cumulative mIoU giảm **-1.46 pp** (41.17% -> 39.71%), Mean Episode-IoU giảm **-1.51 pp** (48.50% -> 46.99%).")
    lines.append("  - *SUIM*: DW3×3 cho kết quả thấp hơn baseline: Cumulative mIoU giảm **-3.21 pp** (38.24% -> 35.03%), Mean Episode-IoU giảm **-5.54 pp** (38.65% -> 33.11%).")
    lines.append("- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Inductive bias mở rộng receptive field từ 1×1 sang 3×3 không mang lại cải thiện đồng đều trên mọi miền dữ liệu. Một giả thuyết khả dĩ là kernel 3×3 với receptive field lớn hơn có thể hữu ích ở các miền có cấu trúc biên rõ (như ảnh giải phẫu hoặc đường sá), nhưng kém phù hợp hơn trên các miền có biên độ tương phản thấp hoặc nhiễu tán xạ cao (như ISIC và SUIM) dưới điều kiện 1-shot SGD trực tuyến. Tuy nhiên, giả thuyết này cần thêm các thực nghiệm kiểm chứng có kiểm soát.")

    # Q2
    lines.append("\n### Q2: Fusion cải tiến (Softmax Margin Fusion) có thực sự hiệu quả không?")
    avg_fus_cum = np.mean(d_fus_cums) if d_fus_cums else 0
    avg_fus_ep = np.mean(d_fus_eps) if d_fus_eps else 0
    lines.append(f"- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **{avg_fus_cum:+.2f} pp** (Mean Episode-IoU: **{avg_fus_ep:+.2f} pp**).")
    lines.append("- **Chi tiết theo từng miền dữ liệu**:")
    lines.append("  - *Lung*: Ghi nhận mức cải thiện rõ nét nhất: Cumulative mIoU tăng **+0.63 pp** (78.69% -> 79.32%), Mean Episode-IoU tăng **+0.61 pp** (78.34% -> 78.95%).")
    lines.append("  - *FSS-1000*: Cumulative mIoU tăng nhẹ **+0.11 pp** (80.08% -> 80.19%), Mean Episode-IoU tăng **+0.14 pp** (80.35% vs 80.21%).")
    lines.append("  - *DeepGlobe*: Cumulative mIoU biến thiên -0.07 pp (50.83% -> 50.76%), Mean Episode-IoU biến thiên -0.24 pp (51.68% -> 51.44%).")
    lines.append("  - *ISIC*: Cumulative mIoU biến thiên -0.21 pp (41.17% -> 40.96%), Mean Episode-IoU biến thiên -0.26 pp (48.50% -> 48.24%).")
    lines.append("  - *SUIM*: Cumulative mIoU biến thiên -0.57 pp (38.24% -> 37.67%), Mean Episode-IoU biến thiên -0.30 pp (38.65% -> 38.35%).")
    lines.append("- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Softmax Margin Fusion điều chỉnh trọng số tầng dựa trên khoảng cách prototype giữa foreground và background. Trên miền Lung, cơ chế này giúp tăng tỷ trọng của các tầng có độ phân tách hình học cao. Trên 4 miền còn lại, kết quả dao động sát mức baseline (biến thiên trung bình toàn benchmark là -0.02 pp).")

    # Q3
    lines.append("\n### Q3: Phương pháp đề xuất kết hợp (E3: DW3×3 + Softmax Margin) có hiệu quả không?")
    avg_comb_cum = np.mean(d_comb_cums) if d_comb_cums else 0
    avg_comb_ep = np.mean(d_comb_eps) if d_comb_eps else 0
    lines.append(f"- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **{avg_comb_cum:+.2f} pp** (Mean Episode-IoU: **{avg_comb_ep:+.2f} pp**).")
    lines.append("- **Chi tiết theo từng miền dữ liệu**:")
    lines.append("  - *Lung*: Cấu hình kết hợp E3 đạt **80.09% Cumulative mIoU** (+1.40 pp so với E0 78.69%) và **79.41% Mean Episode-IoU** (+1.07 pp so với E0 78.34%). Con số 80.09% nằm sát mốc tham chiếu 80.0% được công bố trong bài báo gốc.")
    lines.append("  - *DeepGlobe*: Cumulative mIoU đạt 50.29% (-0.54 pp so với E0), Mean Episode-IoU đạt 51.51% (-0.17 pp so với E0).")
    lines.append("  - *FSS-1000*: Cumulative mIoU đạt 79.60% (-0.48 pp so với E0), Mean Episode-IoU đạt 79.73% (-0.48 pp so với E0).")
    lines.append("  - *ISIC*: Cumulative mIoU đạt 40.52% (-0.65 pp so với E0), Mean Episode-IoU đạt 47.78% (-0.72 pp so với E0).")
    lines.append("  - *SUIM*: Cumulative mIoU đạt 34.75% (-3.49 pp so với E0), Mean Episode-IoU đạt 32.97% (-5.68 pp so với E0).")
    lines.append("- **Kết luận Q3**: Cấu hình kết hợp E3 cải thiện kết quả rõ ràng trên miền Lung (+1.40 pp Cumulative mIoU), nhưng không đem lại cải thiện đồng đều trên toàn bộ 5 benchmark. Hiệu năng trung bình của E3 trên 5 dataset thấp hơn E0 (-0.75 pp Cumulative mIoU, -1.20 pp Mean Episode-IoU), cho thấy hiệu quả của phương pháp kết hợp mang tính phụ thuộc miền (domain-dependent) thay vì ưu việt phổ quát.")

    # Q4
    lines.append("\n### Q4: Có tương tác (interaction) giữa Adapter và Fusion không?")
    if len(d_comb_cums) == len(d_adp_cums) and len(d_comb_cums) > 0:
        diff_interaction = [c - (a + f) for c, a, f in zip(d_comb_cums, d_adp_cums, d_fus_cums)]
        avg_interaction = np.mean(diff_interaction)
        lines.append(f"- **Định lượng tương tác**: Giá trị $\\Delta_{{Combined}} - (\\Delta_{{Adapter}} + \\Delta_{{Fusion}})$ trung bình là **{avg_interaction:+.2f} pp** (Cumulative mIoU).")
        lines.append("  - *Trên Lung*: $\\Delta_{Combined} (+1.40\\text{ pp}) > \\Delta_{Adapter} (+0.15\\text{ pp}) + \\Delta_{Fusion} (+0.63\\text{ pp}) = +0.78\\text{ pp}$. Tương tác quan sát được là **+0.62 pp**, cho thấy kết quả kết hợp trên tập đánh giá này có dạng super-additive.")
        lines.append("  - *Trên các miền còn lại*: Giá trị tương tác dao động: DeepGlobe (-0.47 pp), ISIC (+1.02 pp), FSS-1000 (-0.46 pp), SUIM (+0.29 pp).")
        lines.append("- **Kết luận Q4**: Các kết quả quan sát cho thấy có sự tương tác giữa hai thành phần trên từng tập dữ liệu cụ thể (đặc biệt là Lung), nhưng để khẳng định hiệu ứng cộng hưởng (synergy) có ý nghĩa thống kê tổng quát thì cần mở rộng thêm các thực nghiệm đa seed.")
    else:
        lines.append("- *Đang cập nhật dữ liệu...*")

    lines.append("\n---\n")

    # 7. Reproducibility & Limitations
    lines.append("## 7. Protocol Transparency & Limitations\n")
    lines.append("- **Protocol Control**: Toàn bộ thử nghiệm thực thi cố định với `seed=42`, `nshot=1`, 25 epochs online SGD per episode. Dataset Lung sử dụng manifest 20 episode cố định; 4 dataset còn lại sử dụng runtime sampling có kiểm soát seed.")
    lines.append("- **Zero Tuning**: Các siêu tham số (learning rate 0.01, l0=3, temperature 1.0, threshold max_otsu_mean) được giữ nguyên hoàn toàn xuyên suốt 20 runs.")
    lines.append("- **Limitations**: ")
    lines.append("  1. Quy mô đánh giá gồm 20 episode cho mỗi dataset (do hạn chế tính toán trên CPU), không thay thế cho đánh giá 1.000 episode quy mô lớn trên GPU cluster.")
    lines.append("  2. Thực nghiệm thực hiện trên 1 seed duy nhất (seed=42), chưa đủ để thực hiện kiểm định ý nghĩa thống kê (t-test / ANOVA).")
    lines.append("  3. Các nhận định về nguyên nhân vật lý/hình ảnh (ví dụ: tán xạ dưới nước, sắc tố da) hiện dừng ở mức giả thuyết khoa học hợp lý, cần thêm kiểm chứng phân rã lỗi (error visual breakdown).")

    report_content = "\n".join(lines) + "\n"
    out_path = os.path.join(PROJECT_ROOT, "FULL_BENCHMARK_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[OK] Report written to: {out_path}")
    return out_path

if __name__ == '__main__':
    generate_report()
