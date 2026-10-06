#!/usr/bin/env python3
"""
Full Benchmark Runner for ABCDFSS E0-E3 Evaluation Suite
Runs all 5 benchmark domains across all 4 configurations:
  Datasets: DeepGlobe, ISIC, Lung, FSS-1000, SUIM
  Experiments: E0 (baseline), E1 (adapter), E2 (fusion), E3 (proposed)
  Fixed Protocol: seed=42, nshot=1, 400x400, adapt_to=every-episode, 25 epochs SGD

Key Guarantees:
1. Every benchmark automatically resolves and uses an explicit, deterministic manifest.
2. E0, E1, E2, and E3 use THE EXACT SAME manifest for the same dataset.
3. Cryptographic protocol signature validation prevents invalid resume skips.
4. Historical 20-episode results in results/full_benchmark/ are preserved and never overwritten.
5. Large-scale runs (e.g. 100, 1000, all) route to isolated output directories.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_all_benchmarks import check_or_download_dataset
from src.utils.manifest import resolve_manifest_path, validate_manifest, compute_manifest_sha256
from src.utils.protocol import create_protocol_signature, validate_protocol_signature

BENCHMARK_META = {
    'DeepGlobe': {'benchmark': 'deepglobe', 'slug': 'heyoujue/deepglobe'},
    'ISIC': {'benchmark': 'isic', 'slug': 'heyoujue/isic2018-classwise'},
    'Lung': {'benchmark': 'lung', 'slug': 'heyoujue/lungsegmentation'},
    'FSS1000': {'benchmark': 'fss', 'slug': None},
    'SUIM': {'benchmark': 'suim', 'slug': 'heyoujue/suim-merged'},
}

EXP_DEFINITIONS = {
    'E0': {'adapter': 'conv1x1', 'fusion': 'mean'},
    'E1': {'adapter': 'depthwise_separable_3x3', 'fusion': 'mean'},
    'E2': {'adapter': 'conv1x1', 'fusion': 'softmax_margin'},
    'E3': {'adapter': 'depthwise_separable_3x3', 'fusion': 'softmax_margin'},
}

ALL_EXPERIMENTS = ['E0', 'E1', 'E2', 'E3']

def parse_args():
    parser = argparse.ArgumentParser(description="Full Benchmark Runner for ABCDFSS E0-E3")
    parser.add_argument('--episodes', type=str, default='20',
                        help='Number of episodes per run: 20, 100, 1000, or "all" (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--experiments', nargs='+', default=['E0', 'E1', 'E2', 'E3'],
                        choices=['E0', 'E1', 'E2', 'E3', 'all'],
                        help='List of experiments to run (default: E0 E1 E2 E3)')
    parser.add_argument('--benchmarks', nargs='+', default=['DeepGlobe', 'ISIC', 'Lung', 'FSS1000', 'SUIM'],
                        help='List of benchmark datasets to run')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--nworker', type=int, default=0,
                        help='Dataloader workers (default: 0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom base output directory (default: results/full_benchmark for 20ep, results/full_benchmark_{N}ep for others)')
    return parser.parse_args()

def run_suite(args):
    ep_str = str(args.episodes).lower().strip()

    # Determine safe output directory to isolate runs and preserve historical benchmarks
    if args.output_dir:
        base_results_dir = args.output_dir
    elif ep_str == '20':
        base_results_dir = os.path.join(PROJECT_ROOT, "results", "full_benchmark")
    else:
        suffix = "all_ep" if ep_str == 'all' else f"{ep_str}ep"
        base_results_dir = os.path.join(PROJECT_ROOT, "results", f"full_benchmark_{suffix}")

    os.makedirs(base_results_dir, exist_ok=True)
    experiments_to_run = ALL_EXPERIMENTS if 'all' in args.experiments else args.experiments

    print("=" * 80)
    print("       STARTING FULL CD-FSS BENCHMARK SUITE (E0 - E3 ON 5 DATASETS)")
    print(f"       Episodes: {ep_str} | Seed: {args.seed} | Mode: every-episode | Device: {args.device}")
    print(f"       Output directory: {base_results_dir}")
    print("=" * 80 + "\n")

    # 1. Resolve dataset paths and validate deterministic manifests
    dataset_configs = {}
    for ds_name in args.benchmarks:
        if ds_name in BENCHMARK_META:
            meta = BENCHMARK_META[ds_name]
            resolved_datapath = check_or_download_dataset(meta['benchmark'], kaggle_slug=meta['slug'])

            # Automatically resolve deterministic manifest
            manifest_path = resolve_manifest_path(
                benchmark=meta['benchmark'],
                seed=args.seed,
                episodes=ep_str,
                manifest_path=None,
                allow_missing=False
            )

            # Validate manifest integrity
            manifest_info = validate_manifest(
                manifest_path=manifest_path,
                benchmark=meta['benchmark'],
                seed=args.seed,
                episodes=ep_str,
                dataset_base_path=resolved_datapath
            )

            dataset_configs[ds_name] = {
                'benchmark': meta['benchmark'],
                'datapath': resolved_datapath,
                'manifest_path': manifest_path,
                'manifest_sha256': manifest_info['manifest_sha256'],
                'resolved_episodes': manifest_info['resolved_episodes']
            }

    total_runs = len(dataset_configs) * len(experiments_to_run)
    current_run = 0

    for ds_name, ds_info in dataset_configs.items():
        # Exact same manifest is used across E0, E1, E2, E3
        manifest_path = ds_info['manifest_path']
        manifest_sha256 = ds_info['manifest_sha256']
        actual_episodes = ds_info['resolved_episodes']

        for exp in experiments_to_run:
            current_run += 1
            exp_def = EXP_DEFINITIONS[exp]
            exp_dir = os.path.join(base_results_dir, ds_name, exp)
            os.makedirs(exp_dir, exist_ok=True)
            result_json = os.path.join(exp_dir, "run_result.json")

            # Construct expected protocol signature for strong resume validation
            expected_sig = create_protocol_signature(
                benchmark=ds_info['benchmark'],
                experiment=exp,
                episodes=actual_episodes,
                seed=args.seed,
                nshot=1,
                adapt_to="every-episode",
                adapter=exp_def['adapter'],
                fusion=exp_def['fusion'],
                fusion_temp=1.0,
                image_size=400,
                num_epochs=25,
                learning_rate=0.01,
                manifest=manifest_path,
                manifest_sha256=manifest_sha256
            )

            # Strong Resume Check: skip ONLY if complete protocol signature matches
            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)

                    saved_sig = saved_data.get('protocol_signature')
                    is_valid_resume, reason = validate_protocol_signature(saved_sig, expected_sig)

                    if is_valid_resume:
                        cum_miou = saved_data['metric']['Cumulative_mIoU']
                        mean_ep_iou = saved_data['metric']['Mean_Episode_IoU']
                        print(f"[{current_run:02d}/{total_runs:02d}] [SKIP - SIGNATURE VERIFIED] {ds_name} {exp} => Cumulative mIoU: {cum_miou:.2f}%, Mean Episode-IoU: {mean_ep_iou:.2f}%")
                        continue
                    else:
                        print(f"[{current_run:02d}/{total_runs:02d}] [RE-RUN REQUIRED] {ds_name} {exp}: {reason}")
                except Exception as e:
                    print(f"[{current_run:02d}/{total_runs:02d}] [RE-RUN REQUIRED] Could not verify existing result: {e}")

            print(f"\n{'='*70}")
            print(f"[{current_run:02d}/{total_runs:02d}] EXECUTING: {ds_name} - {exp}")
            print(f"Datapath:    {ds_info['datapath']}")
            print(f"Manifest:    {manifest_path} (SHA256: {manifest_sha256[:12]}...)")
            print(f"Output dir:  {exp_dir}")
            print(f"{'='*70}\n")

            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "evaluate.py"),
                "--benchmark", ds_info['benchmark'],
                "--datapath", ds_info['datapath'],
                "--experiment", exp,
                "--episodes", ep_str,
                "--manifest", manifest_path,
                "--seed", str(args.seed),
                "--nshot", "1",
                "--adapt-to", "every-episode",
                "--device", args.device,
                "--nworker", str(args.nworker),
                "--logpath", exp_dir
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = PROJECT_ROOT + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

            run_start = time.time()
            res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            run_time = time.time() - run_start

            if res.returncode != 0:
                print(f"[!] ERROR: Run failed for {ds_name} {exp} with returncode {res.returncode}")
            else:
                print(f"[OK] Completed {ds_name} {exp} in {run_time:.1f}s")

    print("\n\n" + "=" * 80)
    print("              FULL BENCHMARK EXECUTION COMPLETED")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    args = parse_args()
    run_suite(args)
