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

    # 1. Audit Protocol Signatures & Manifest Hash
    sig1 = res1.get('protocol_signature', res1.get('config', {}))
    sig2 = res2.get('protocol_signature', res2.get('config', {}))

    print("\n[1] Protocol Signature & Manifest Integrity Audit:")
    is_compat, reason = validate_protocol_signature(sig1, sig2)
    if is_compat:
        print("    [PASS] Protocol signatures match identically.")
    else:
        print(f"    [WARN] Protocol signature difference: {reason}")

    sha1 = sig1.get('manifest_sha256') or res1.get('config', {}).get('manifest_sha256')
    sha2 = sig2.get('manifest_sha256') or res2.get('config', {}).get('manifest_sha256')
    manifest_match = True
    if sha1 and sha2:
        if sha1 != sha2:
            manifest_match = False
            print(f"    [FAIL] Manifest SHA256 mismatch: run1={sha1} vs run2={sha2}")
        else:
            print(f"    [PASS] Manifest SHA256 verified identically: {sha1}")
    elif sha1 or sha2:
        manifest_match = False
        print(f"    [FAIL] Asymmetric manifest SHA256 (run1: {sha1}, run2: {sha2})")

    # Extract Environment Telemetry
    env1 = res1.get('environment', {})
    env2 = res2.get('environment', {})
    print(f"    Run 1 Environment: Device={env1.get('gpu_name', 'N/A')}, PyTorch={env1.get('torch_version', 'N/A')}")
    print(f"    Run 2 Environment: Device={env2.get('gpu_name', 'N/A')}, PyTorch={env2.get('torch_version', 'N/A')}")

    # 2. Episode Set & Count Pre-Validation (by episode_id)
    def extract_episodes_map(res: dict):
        raw = res.get('raw_result', {})
        detailed = raw.get('detailed_episodes', [])
        if detailed:
            return {int(ep['episode_id']): float(ep['iou']) for ep in detailed}, len(detailed)
        ious = raw.get('episode_ious', [])
        return {int(idx): float(iou) for idx, iou in enumerate(ious)}, len(ious)

    ep_map1, count1 = extract_episodes_map(res1)
    ep_map2, count2 = extract_episodes_map(res2)

    count_match = (count1 == count2 and count1 > 0)
    if not count_match:
        print(f"\n[2] Episode Alignment Pre-Check:")
        print(f"    [FAIL] Episode count mismatch: run1 has {count1} episodes, run2 has {count2} episodes.")
    else:
        print(f"\n[2] Episode Alignment Pre-Check:")
        print(f"    [PASS] Episode count identical: {count1} episodes.")

    ids1 = set(ep_map1.keys())
    ids2 = set(ep_map2.keys())
    ids_match = (ids1 == ids2 and len(ids1) > 0)
    if not ids_match:
        diff_ids = ids1.symmetric_difference(ids2)
        print(f"    [FAIL] Episode ID set mismatch. Symmetric difference count: {len(diff_ids)}")
    else:
        print(f"    [PASS] Episode ID sets match exactly ({len(ids1)} verified IDs).")

    # 3. Summary Metric Consistency
    m1 = res1.get('metric', {})
    m2 = res2.get('metric', {})

    mean_ep1 = m1.get('Mean_Episode_IoU', 0.0)
    mean_ep2 = m2.get('Mean_Episode_IoU', 0.0)
    diff_mean_ep = abs(mean_ep1 - mean_ep2)

    cum1 = m1.get('Cumulative_mIoU', 0.0)
    cum2 = m2.get('Cumulative_mIoU', 0.0)
    diff_cum = abs(cum1 - cum2)

    print("\n[3] Summary Metric Consistency:")
    print(f"    Mean Episode-IoU:  Run1={mean_ep1:.2f}%, Run2={mean_ep2:.2f}%  => Delta = {diff_mean_ep:.4f}% (Tol: {tol_mean}%)")
    print(f"    Cumulative mIoU:   Run1={cum1:.2f}%, Run2={cum2:.2f}%  => Delta = {diff_cum:.4f}% (Tol: {tol_mean}%)")

    # 4. Episode-Level Granular Audit (mapped strictly by episode_id)
    common_ids = sorted(ids1.intersection(ids2))
    if not common_ids:
        print("    [!] No common episode IDs between run1 and run2 to compute deltas.")
        ep_pass = False
        max_ep_diff = float('inf')
        mean_ep_diff = float('inf')
    else:
        deltas = [abs(ep_map1[eid] - ep_map2[eid]) for eid in common_ids]
        max_ep_diff = max(deltas)
        mean_ep_diff = float(np.mean(deltas))

        print(f"\n[4] Granular Episode Audit (across {len(common_ids)} episodes mapped strictly by episode_id):")
        print(f"    Mean Abs Episode Delta: {mean_ep_diff:.4f}%")
        print(f"    Max Abs Episode Delta:  {max_ep_diff:.4f}% (Tol: {tol_max}%)")
        print("    [NOTE] Specified tolerances represent metric-level numerical tolerances")
        print("           (accounting for floating-point arithmetic / non-deterministic GPU kernel reductions),")
        print("           NOT bitwise binary equality.")
        ep_pass = (max_ep_diff <= tol_max)

    mean_pass = (diff_mean_ep <= tol_mean) and (diff_cum <= tol_mean)
    overall_pass = is_compat and manifest_match and count_match and ids_match and mean_pass and ep_pass

    print("\n" + "=" * 80)
    print(f"  REPRODUCIBILITY VERDICT: {'[PASS]' if overall_pass else '[FAIL]'}")
    print("=" * 80 + "\n")
    return {
        "overall_pass": overall_pass,
        "signature_match": is_compat,
        "manifest_match": manifest_match,
        "count_match": count_match,
        "ids_match": ids_match,
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
