#!/usr/bin/env python3
"""
Full Benchmark Runner for ABCDFSS E0-E3 Evaluation Suite
Runs all 5 benchmark domains across all 4 configurations:
  Datasets: DeepGlobe, ISIC, Lung, FSS-1000, SUIM
  Experiments: E0 (baseline), E1 (adapter), E2 (fusion), E3 (proposed)
  Fixed Protocol: seed=42, nshot=1, 400x400, adapt_to=every-episode, 25 epochs SGD

Preserves historical 20-episode results in:
  results/full_benchmark/{Dataset}/{Experiment}/
Saves larger scale results (e.g. 100 or 1000 episodes) in:
  results/full_benchmark_{episodes}ep/{Dataset}/{Experiment}/
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

BENCHMARK_META = {
    'DeepGlobe': {'benchmark': 'deepglobe', 'slug': 'heyoujue/deepglobe'},
    'ISIC': {'benchmark': 'isic', 'slug': 'heyoujue/isic2018-classwise'},
    'Lung': {'benchmark': 'lung', 'slug': 'heyoujue/lungsegmentation'},
    'FSS1000': {'benchmark': 'fss', 'slug': None},
    'SUIM': {'benchmark': 'suim', 'slug': 'heyoujue/suim-merged'},
}

ALL_EXPERIMENTS = ['E0', 'E1', 'E2', 'E3']

def parse_args():
    parser = argparse.ArgumentParser(description="Full Benchmark Runner for ABCDFSS E0-E3")
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes per run (default: 20)')
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
    # Determine safe output directory that preserves historical results
    if args.output_dir:
        base_results_dir = args.output_dir
    elif args.episodes == 20:
        base_results_dir = os.path.join(PROJECT_ROOT, "results", "full_benchmark")
    else:
        base_results_dir = os.path.join(PROJECT_ROOT, "results", f"full_benchmark_{args.episodes}ep")

    os.makedirs(base_results_dir, exist_ok=True)

    experiments_to_run = ALL_EXPERIMENTS if 'all' in args.experiments else args.experiments

    print("=" * 80)
    print("       STARTING FULL CD-FSS BENCHMARK SUITE (E0 - E3 ON 5 DATASETS)")
    print(f"       Episodes: {args.episodes} | Seed: {args.seed} | Mode: every-episode | Device: {args.device}")
    print(f"       Output directory: {base_results_dir}")
    print("=" * 80 + "\n")

    # Resolve dataset paths dynamically
    dataset_configs = {}
    for ds_name in args.benchmarks:
        if ds_name in BENCHMARK_META:
            meta = BENCHMARK_META[ds_name]
            resolved_path = check_or_download_dataset(meta['benchmark'], kaggle_slug=meta['slug'])
            dataset_configs[ds_name] = {
                'benchmark': meta['benchmark'],
                'datapath': resolved_path
            }

    total_runs = len(dataset_configs) * len(experiments_to_run)
    current_run = 0

    for ds_name, ds_info in dataset_configs.items():
        for exp in experiments_to_run:
            current_run += 1
            exp_dir = os.path.join(base_results_dir, ds_name, exp)
            os.makedirs(exp_dir, exist_ok=True)
            result_json = os.path.join(exp_dir, "run_result.json")

            # Check if run already completed with matching episode count
            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get('episode_count') == args.episodes:
                        cum_miou = data['metric']['Cumulative_mIoU']
                        mean_ep_iou = data['metric']['Mean_Episode_IoU']
                        print(f"[{current_run:02d}/{total_runs:02d}] [SKIP - ALREADY COMPLETED] {ds_name} {exp} => Cumulative mIoU: {cum_miou:.2f}%, Mean Episode-IoU: {mean_ep_iou:.2f}%")
                        continue
                except Exception:
                    pass

            print(f"\n{'='*70}")
            print(f"[{current_run:02d}/{total_runs:02d}] EXECUTING: {ds_name} - {exp}")
            print(f"Datapath: {ds_info['datapath']}")
            print(f"Output directory: {exp_dir}")
            print(f"{'='*70}\n")

            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "evaluate.py"),
                "--benchmark", ds_info['benchmark'],
                "--datapath", ds_info['datapath'],
                "--experiment", exp,
                "--episodes", str(args.episodes),
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
