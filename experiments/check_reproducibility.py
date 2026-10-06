#!/usr/bin/env python3
"""
CPU/GPU and Cross-Device Reproducibility Verification Utility.
Audits numerical consistency, protocol signatures, and per-episode IoU tolerances
between two benchmark execution artifacts (e.g. CPU vs GPU, or local vs cloud Modal).

Usage:
    python experiments/check_reproducibility.py --run1 results/.../run_result.json --run2 results/.../run_result.json
    python experiments/check_reproducibility.py --tolerance-mean 0.15 --tolerance-max 0.50 --run1 ... --run2 ...
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.protocol import validate_protocol_signature

DEFAULT_MEAN_TOLERANCE = 0.15  # Max allowed difference in Mean Episode-IoU / Cumulative mIoU (in %)
DEFAULT_MAX_EP_TOLERANCE = 0.50 # Max allowed difference on any single episode IoU (in %)

def compare_runs(
    run1_path: str,
    run2_path: str,
    tol_mean: float = DEFAULT_MEAN_TOLERANCE,
    tol_max: float = DEFAULT_MAX_EP_TOLERANCE
):
    print("=" * 80)
    print("           ABCDFSS CROSS-ENVIRONMENT REPRODUCIBILITY AUDIT")
    print("=" * 80)
    print(f"Run 1 Artifact: {run1_path}")
    print(f"Run 2 Artifact: {run2_path}")
    print(f"Tolerances: Mean IoU <= {tol_mean:.2f}%, Single-Episode IoU <= {tol_max:.2f}%")
    print("-" * 80)

    if not os.path.exists(run1_path):
        raise FileNotFoundError(f"Run 1 artifact does not exist: {run1_path}")
    if not os.path.exists(run2_path):
        raise FileNotFoundError(f"Run 2 artifact does not exist: {run2_path}")

    with open(run1_path, 'r', encoding='utf-8') as f:
        res1 = json.load(f)
    with open(run2_path, 'r', encoding='utf-8') as f:
        res2 = json.load(f)

    # 1. Audit Protocol Signatures
    sig1 = res1.get('protocol_signature', res1.get('config', {}))
    sig2 = res2.get('protocol_signature', res2.get('config', {}))

    print("\n[1] Protocol Signature Audit:")
    is_compat, reason = validate_protocol_signature(sig1, sig2)
    if is_compat:
        print("    [PASS] Protocol signatures match identically.")
    else:
        print(f"    [WARN] Protocol signature difference: {reason}")

    # Extract Environment Telemetry
    env1 = res1.get('environment', {})
    env2 = res2.get('environment', {})
    print(f"    Run 1 Environment: Device={env1.get('gpu_name', 'N/A')}, PyTorch={env1.get('torch_version', 'N/A')}")
    print(f"    Run 2 Environment: Device={env2.get('gpu_name', 'N/A')}, PyTorch={env2.get('torch_version', 'N/A')}")

    # 2. Metric Comparisons
    m1 = res1.get('metric', {})
    m2 = res2.get('metric', {})

    mean_ep1 = m1.get('Mean_Episode_IoU', 0.0)
    mean_ep2 = m2.get('Mean_Episode_IoU', 0.0)
    diff_mean_ep = abs(mean_ep1 - mean_ep2)

    cum1 = m1.get('Cumulative_mIoU', 0.0)
    cum2 = m2.get('Cumulative_mIoU', 0.0)
    diff_cum = abs(cum1 - cum2)

    print("\n[2] Summary Metric Consistency:")
    print(f"    Mean Episode-IoU:  Run1={mean_ep1:.2f}%, Run2={mean_ep2:.2f}%  => Delta = {diff_mean_ep:.4f}% (Tol: {tol_mean}%)")
    print(f"    Cumulative mIoU:   Run1={cum1:.2f}%, Run2={cum2:.2f}%  => Delta = {diff_cum:.4f}% (Tol: {tol_mean}%)")

    # 3. Episode-Level Granular Audit
    ious1 = res1.get('raw_result', {}).get('episode_ious', [])
    ious2 = res2.get('raw_result', {}).get('episode_ious', [])

    if len(ious1) == 0 or len(ious2) == 0:
        print("    [!] Missing raw episode IoUs in one or both runs.")
        ep_pass = False
        max_ep_diff = 0.0
    else:
        count = min(len(ious1), len(ious2))
        deltas = [abs(ious1[i] - ious2[i]) for i in range(count)]
        max_ep_diff = max(deltas)
        mean_ep_diff = float(np.mean(deltas))

        print(f"\n[3] Granular Episode Audit (across {count} episodes):")
        print(f"    Mean Abs Episode Delta: {mean_ep_diff:.4f}%")
        print(f"    Max Abs Episode Delta:  {max_ep_diff:.4f}% (Tol: {tol_max}%)")
        ep_pass = (max_ep_diff <= tol_max)

    mean_pass = (diff_mean_ep <= tol_mean) and (diff_cum <= tol_mean)
    overall_pass = is_compat and mean_pass and ep_pass

    print("\n" + "=" * 80)
    print(f"  REPRODUCIBILITY VERDICT: {'[PASS]' if overall_pass else '[FAIL]'}")
    print("=" * 80 + "\n")
    return {
        "overall_pass": overall_pass,
        "signature_match": is_compat,
        "delta_mean_episode_iou": diff_mean_ep,
        "delta_cumulative_miou": diff_cum,
        "max_episode_delta": max_ep_diff
    }

def main():
    parser = argparse.ArgumentParser(description="Cross-Device Reproducibility Verification Utility")
    parser.add_argument('--run1', type=str, required=True, help='Path to first run_result.json')
    parser.add_argument('--run2', type=str, required=True, help='Path to second run_result.json')
    parser.add_argument('--tolerance-mean', type=float, default=DEFAULT_MEAN_TOLERANCE,
                        help=f'Tolerance for Mean Episode-IoU and Cumulative mIoU (default: {DEFAULT_MEAN_TOLERANCE} pct)')
    parser.add_argument('--tolerance-max', type=float, default=DEFAULT_MAX_EP_TOLERANCE,
                        help=f'Tolerance for single-episode max IoU difference (default: {DEFAULT_MAX_EP_TOLERANCE} pct)')
    args = parser.parse_args()

    res = compare_runs(args.run1, args.run2, tol_mean=args.tolerance_mean, tol_max=args.tolerance_max)
    sys.exit(0 if res["overall_pass"] else 1)

if __name__ == '__main__':
    main()
