import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

benchmarks = ['deepglobe', 'isic', 'lung', 'fss', 'suim']
experiments = ['E0', 'E1', 'E2', 'E3']

print("=" * 90)
print(f"{'Benchmark':<12} | {'Exp':<5} | {'Mean Ep IoU':<13} | {'Cum mIoU':<12} | {'FB-IoU':<10} | {'Episodes':<8} | {'Manifest Hash':<16}")
print("-" * 90)

paper_miou = {
    'deepglobe': 42.30,
    'isic': 41.80,
    'lung': 80.00,
    'fss': 69.30,
    'suim': 35.00
}

summary_table = {}

for b in benchmarks:
    summary_table[b] = {}
    for exp in experiments:
        path = os.path.join(PROJECT_ROOT, "results", b, f"{exp}_1000ep_seed42", "run_result.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = data.get("metric", {})
            mean_ep = m.get("Mean_Episode_IoU", 0.0)
            cum_iou = m.get("Cumulative_mIoU", 0.0)
            fb_iou = m.get("FB_IoU", 0.0)
            ep_cnt = data.get("episode_count", len(data.get("raw_result", {}).get("episode_ious", [])))
            sig = data.get("protocol_signature", {})
            m_hash = sig.get("manifest_sha256", "")[:12] + "..." if sig.get("manifest_sha256") else "N/A"
            summary_table[b][exp] = {
                "mean_ep": mean_ep,
                "cum_iou": cum_iou,
                "fb_iou": fb_iou,
                "count": ep_cnt
            }
            print(f"{b:<12} | {exp:<5} | {mean_ep:>10.2f}%   | {cum_iou:>9.2f}%   | {fb_iou:>7.2f}%  | {ep_cnt:<8} | {m_hash:<16}")
        else:
            print(f"{b:<12} | {exp:<5} | MISSING!")

print("=" * 90)
print("\n" + "=" * 90)
print("              COMPREHENSIVE SCIENTIFIC BENCHMARK MATRIX (1000 EPISODES)")
print("=" * 90)
print(f"{'Dataset':<12} | {'Paper (CVPR)':<14} | {'E0 (Baseline)':<14} | {'E1 (DW+Mean)':<14} | {'E2 (1x1+Margin)':<16} | {'E3 (DW+Margin)':<16}")
print("-" * 90)
for b in benchmarks:
    p_val = f"{paper_miou.get(b, 0.0):.2f}%"
    e0_val = f"{summary_table[b].get('E0', {}).get('cum_iou', 0.0):.2f}%"
    e1_val = f"{summary_table[b].get('E1', {}).get('cum_iou', 0.0):.2f}%"
    e2_val = f"{summary_table[b].get('E2', {}).get('cum_iou', 0.0):.2f}%"
    e3_val = f"{summary_table[b].get('E3', {}).get('cum_iou', 0.0):.2f}%"
    print(f"{b.upper():<12} | {p_val:<14} | {e0_val:<14} | {e1_val:<14} | {e2_val:<16} | {e3_val:<16}")
print("=" * 90)
