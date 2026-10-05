#!/usr/bin/env python3
"""
Full Benchmark Runner for ABCDFSS E0-E3 Evaluation Suite
Runs all 5 benchmark domains across all 4 configurations:
  Datasets: DeepGlobe, ISIC, Lung, FSS-1000, SUIM
  Experiments: E0 (baseline), E1 (adapter), E2 (fusion), E3 (proposed)
  Fixed Protocol: seed=42, nshot=1, 400x400, adapt_to=every-episode, 25 epochs SGD
Saves individual structured artifacts into:
  results/full_benchmark/{Dataset}/{Experiment}/
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIGS = {
    'DeepGlobe': {
        'benchmark': 'deepglobe',
        'datapath': os.path.join(PROJECT_ROOT, 'datasets', 'deepglobe')
    },
    'ISIC': {
        'benchmark': 'isic',
        'datapath': r"C:\Users\TUF DASH\.cache\kagglehub\datasets\heyoujue\isic2018-classwise\versions\1"
    },
    'Lung': {
        'benchmark': 'lung',
        'datapath': r"C:\Users\TUF DASH\.cache\kagglehub\datasets\heyoujue\lungsegmentation\versions\2"
    },
    'FSS1000': {
        'benchmark': 'fss',
        'datapath': os.path.join(PROJECT_ROOT, 'datasets', 'fss1000')
    },
    'SUIM': {
        'benchmark': 'suim',
        'datapath': r"C:\Users\TUF DASH\.cache\kagglehub\datasets\heyoujue\suim-merged\versions\2"
    }
}

EXPERIMENTS = ['E0', 'E1', 'E2', 'E3']

def run_suite(episodes: int = 20, seed: int = 42):
    base_results_dir = os.path.join(PROJECT_ROOT, "results", "full_benchmark")
    os.makedirs(base_results_dir, exist_ok=True)

    print("=" * 80)
    print("       STARTING FULL CD-FSS BENCHMARK SUITE (E0 - E3 ON 5 DATASETS)")
    print(f"       Episodes per run: {episodes} | Seed: {seed} | Mode: every-episode")
    print("=" * 80 + "\n")

    total_runs = len(DATASET_CONFIGS) * len(EXPERIMENTS)
    current_run = 0

    for ds_name, ds_info in DATASET_CONFIGS.items():
        for exp in EXPERIMENTS:
            current_run += 1
            exp_dir = os.path.join(base_results_dir, ds_name, exp)
            os.makedirs(exp_dir, exist_ok=True)
            result_json = os.path.join(exp_dir, "run_result.json")

            # Check if run already completed
            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get('episode_count') == episodes:
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
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--nshot", "1",
                "--adapt-to", "every-episode",
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
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_suite(episodes=episodes)
