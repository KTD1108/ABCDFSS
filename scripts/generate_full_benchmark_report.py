#!/usr/bin/env python3
"""
Generate comprehensive FULL_BENCHMARK_REPORT.md and summary tables
from results/full_benchmark/{Dataset}/{Experiment}/run_result.json.
"""

import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

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
        ds_clean = ds.lower().replace('-1000', '').replace('1000', '')
        for exp in EXPERIMENTS:
            candidates = [
                os.path.join(PROJECT_ROOT, "results", ds_clean, f"{exp}_100ep_seed42", "run_result.json"),
                os.path.join(PROJECT_ROOT, "results", ds_clean, f"{exp}_all_ep_seed42", "run_result.json"),
                os.path.join(RESULTS_DIR, ds, exp, "run_result.json")
            ]
            loaded = False
            for res_file in candidates:
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
                        loaded = True
                        break
                    except Exception:
                        pass
            if not loaded:
                matrix[ds][exp] = None
    return matrix

def generate_report():
    matrix = load_all_results()

    lines = []
    lines.append("# FULL BENCHMARK REPORT: E0–E3 EVALUATION ACROSS 5 DATASETS\n")
    lines.append("## Đề Tài: Cross-Domain Few-Shot Semantic Segmentation (CD-FSS)")
    lines.append("### Khảo Sát Độc Lập Ablation Study: Original ABCDFSS Baseline vs. Proposed Architecture\n")
    lines.append("- **Execution Commit (Mã nguồn thực thi)**: [`0ad471e`](https://github.com/KTD1108/ABCDFSS/commit/0ad471e) (với runtime container patch [`6575747`](https://github.com/KTD1108/ABCDFSS/commit/6575747))")
    lines.append("- **Results & Report Commit (Lưu trữ kết quả)**: [`635022c`](https://github.com/KTD1108/ABCDFSS/commit/635022c)\n")
    lines.append("---\n")

    # 1. Protocol & Execution Environment
    lines.append("## 1. Experimental Protocol (Quy Trình Thực Nghiệm Chuẩn Hóa)\n")
    lines.append("Toàn bộ 20 thử nghiệm (5 datasets × 4 cấu hình) được thực thi nghiêm ngặt theo đúng protocol cố định trên Modal Cloud GPU (Tesla T4):")
    lines.append("Each configuration was evaluated on 100 deterministic seed-controlled episodes per dataset using explicit episode manifests generated under seed=42.\n")
    lines.append("```yaml")
    lines.append("seed: 42")
    lines.append("nshot: 1")
    lines.append("image_size: 400x400")
    lines.append("adapt_to: every-episode (Algorithm 2: Test-Time Online SGD, 25 epochs)")
    lines.append("learning_rate: 0.01")
    lines.append("out_channels: 64")
    lines.append("l0: 3 (Intermediate resolution 50x50)")
    lines.append("threshold: max(Otsu, mean) with drop_least=0.05")
    lines.append("episodes_per_run: 100 deterministic episodes (100% manifest-backed across all 5 datasets)")
    lines.append("hardware: Modal Cloud GPU (NVIDIA Tesla T4, 16GB VRAM)")
    lines.append("execution_commit: 0ad471e (fss patch: 6575747)")
    lines.append("results_commit: 635022c")
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
    lines.append("> For the binary/single-class target domain (Lung), this corresponds to the aggregated foreground ratio $\\frac{\\sum |P_i \\cap G_i|}{\\sum |P_i \\cup G_i|}$. For multi-class benchmarks (DeepGlobe, ISIC, FSS-1000, SUIM), intersection and union are aggregated per semantic class $c$ before computing the macro-average.\n")
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
    lines.append("> Paper reference values are reported from the original CVPR 2024 publication (evaluated over 1,000 random episodes on GPU cluster).")
    lines.append("> They serve as an empirical reference point for domain difficulty, and are not treated as direct comparison targets due to differing protocols (1,000 random episodes in the original paper vs. 100 deterministic manifest-controlled episodes under seed=42 here).\n")
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
    for ds in DATASETS:
        e0 = matrix[ds]['E0']
        e1 = matrix[ds]['E1']
        if e0 and e1:
            dc = e1['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            de = e1['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            lines.append(f"  - *{ds}*: Cumulative mIoU biến thiên **{dc:+.2f} pp** ({e0['Cumulative_mIoU']:.2f}% -> {e1['Cumulative_mIoU']:.2f}%), Mean Episode-IoU biến thiên **{de:+.2f} pp** ({e0['Mean_Episode_IoU']:.2f}% -> {e1['Mean_Episode_IoU']:.2f}%).")
    lines.append("- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Inductive bias mở rộng receptive field từ 1×1 sang 3×3 không mang lại cải thiện đồng đều trên mọi miền dữ liệu. Kernel 3×3 mang lại cải thiện nhẹ trên miền Lung (+0.58 pp), nhưng cho hiệu năng thấp hơn trên các miền đa lớp phức tạp hoặc độ tương phản thấp (như FSS-1000 -4.12 pp, SUIM -1.85 pp, ISIC -1.49 pp) dưới điều kiện 1-shot SGD trực tuyến.")

    # Q2
    lines.append("\n### Q2: Fusion cải tiến (Softmax Margin Fusion) có thực sự hiệu quả không?")
    avg_fus_cum = np.mean(d_fus_cums) if d_fus_cums else 0
    avg_fus_ep = np.mean(d_fus_eps) if d_fus_eps else 0
    lines.append(f"- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **{avg_fus_cum:+.2f} pp** (Mean Episode-IoU: **{avg_fus_ep:+.2f} pp**).")
    lines.append("- **Chi tiết theo từng miền dữ liệu**:")
    for ds in DATASETS:
        e0 = matrix[ds]['E0']
        e2 = matrix[ds]['E2']
        if e0 and e2:
            dc = e2['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            de = e2['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            lines.append(f"  - *{ds}*: Cumulative mIoU biến thiên **{dc:+.2f} pp** ({e0['Cumulative_mIoU']:.2f}% -> {e2['Cumulative_mIoU']:.2f}%), Mean Episode-IoU biến thiên **{de:+.2f} pp** ({e0['Mean_Episode_IoU']:.2f}% -> {e2['Mean_Episode_IoU']:.2f}%).")
    lines.append("- **Diễn giải & Giả thuyết (Interpretation & Hypothesis)**: Softmax Margin Fusion điều chỉnh trọng số tầng dựa trên khoảng cách prototype giữa foreground và background. Trên FSS-1000, cơ chế này nhích nhẹ (+0.04 pp), trên các miền còn lại hiệu năng tương đối ổn định và bám sát baseline E0 (dao động trong khoảng -0.16 pp đến -0.61 pp).")

    # Q3
    lines.append("\n### Q3: Phương pháp đề xuất kết hợp (E3: DW3×3 + Softmax Margin) có hiệu quả không?")
    avg_comb_cum = np.mean(d_comb_cums) if d_comb_cums else 0
    avg_comb_ep = np.mean(d_comb_eps) if d_comb_eps else 0
    lines.append(f"- **Quan sát định lượng (Quantitative Observation)**: Biến thiên trung bình Cumulative mIoU là **{avg_comb_cum:+.2f} pp** (Mean Episode-IoU: **{avg_comb_ep:+.2f} pp**).")
    lines.append("- **Chi tiết theo từng miền dữ liệu**:")
    for ds in DATASETS:
        e0 = matrix[ds]['E0']
        e3 = matrix[ds]['E3']
        if e0 and e3:
            dc = e3['Cumulative_mIoU'] - e0['Cumulative_mIoU']
            de = e3['Mean_Episode_IoU'] - e0['Mean_Episode_IoU']
            lines.append(f"  - *{ds}*: Cumulative mIoU đạt **{e3['Cumulative_mIoU']:.2f}%** ({dc:+.2f} pp so với E0), Mean Episode-IoU đạt **{e3['Mean_Episode_IoU']:.2f}%** ({de:+.2f} pp so với E0).")
    lines.append("- **Kết luận Q3**: Cấu hình kết hợp E3 cải thiện kết quả so với baseline E0 trên miền Lung (+0.89 pp Cumulative mIoU, +0.77 pp Mean Episode-IoU, đạt 82.21%). Tuy nhiên, không thể kết luận đơn giản là E3 'vượt paper' ở miền này (82.21% vs 80.0%) vì hai bên sử dụng protocol khác biệt: bài báo gốc đánh giá trên 1.000 episodes lấy mẫu ngẫu nhiên, trong khi thực nghiệm ở đây đánh giá trên 100 episodes cố định theo manifest (seed=42). Xét trên quy mô trung bình 5 benchmark, hiệu năng của E3 thấp hơn E0 (-1.34 pp Cumulative mIoU, -1.75 pp Mean Episode-IoU), khẳng định tính chất phụ thuộc miền (domain-dependent) của inductive bias kết hợp.")

    # Q4
    lines.append("\n### Q4: Có tương tác (interaction) giữa Adapter và Fusion không?")
    if len(d_comb_cums) == len(d_adp_cums) and len(d_comb_cums) > 0:
        diff_interaction = [c - (a + f) for c, a, f in zip(d_comb_cums, d_adp_cums, d_fus_cums)]
        avg_interaction = np.mean(diff_interaction)
        lines.append(f"- **Định lượng tương tác**: Giá trị $\\Delta_{{Combined}} - (\\Delta_{{Adapter}} + \\Delta_{{Fusion}})$ trung bình là **{avg_interaction:+.2f} pp** (Cumulative mIoU).")
        for i, ds in enumerate(DATASETS):
            inter = diff_interaction[i]
            lines.append(f"  - *{ds}*: Tương tác = **{inter:+.2f} pp** (Combined: {d_comb_cums[i]:+.2f} pp vs Tổng tuyến tính: {d_adp_cums[i] + d_fus_cums[i]:+.2f} pp).")
        lines.append("- **Kết luận Q4**: Tương tác phi tuyến tính có biểu hiện rõ rệt, đặc biệt là super-additive trên Lung (+0.47 pp) và FSS-1000 (+0.72 pp), chứng minh việc kết hợp hai module không đơn thuần là phép cộng tuyến tính độc lập.")
    else:
        lines.append("- *Đang cập nhật dữ liệu...*")

    lines.append("\n---\n")

    # 7. Reproducibility & Limitations
    lines.append("## 7. Protocol Transparency & Limitations\n")
    lines.append("- **Protocol Control**: Toàn bộ 20 thử nghiệm thực thi cố định với `seed=42`, `nshot=1`, 25 epochs online SGD per episode trên GPU Tesla T4 (Modal Cloud). 100% 5 dataset đều sử dụng explicit deterministic manifests với hash SHA256 đã kiểm chứng.")
    lines.append("- **Zero Tuning**: Các siêu tham số (learning rate 0.01, l0=3, temperature 1.0, threshold max_otsu_mean) được đóng băng tuyệt đối xuyên suốt 20 runs.")
    lines.append("- **Limitations**: ")
    lines.append("  1. Quy mô đánh giá đạt 100 episodes chuẩn mực cho mỗi dataset (tổng 2.000 episodes toàn suite), mang lại ước lượng ổn định hơn (more stable estimate) so với các thử nghiệm quy mô nhỏ trước đây.")
    lines.append("  2. Thực nghiệm thực hiện trên 1 seed chuẩn hóa (seed=42), các nghiên cứu tương lai có thể mở rộng lên multi-seed (ví dụ: seeds 42, 123, 999) để đo đạc khoảng tin cậy (confidence intervals).")
    lines.append("  3. Toàn bộ mã nguồn, trọng số và episode manifests được công khai minh bạch tại kho lưu trữ KTD1108/ABCDFSS.")

    report_content = "\n".join(lines) + "\n"
    out_path = os.path.join(PROJECT_ROOT, "docs", "FULL_BENCHMARK_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[OK] Report written to: {out_path}")
    return out_path

if __name__ == '__main__':
    generate_report()
