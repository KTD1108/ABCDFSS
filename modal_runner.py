#!/usr/bin/env python3
"""
Modal Serverless GPU Runner for CD-FSS Benchmarking
Runs evaluation directly on cloud GPU (T4 / A10G / A100) matching the CVPR 2024 exact protocol:
  Algorithm 2: --adapt-to every-episode --episodes 1000 (or 20, 100)

Features:
- Persistent Modal Volume for dataset caching across runs (/root/datasets_cache)
- Persistent Modal Volume for evaluation logs and artifacts (/root/results)
- Configurable GPU selection (T4, A10G, A100)
- Support for single experiments (E0, E1, E2, E3) or matrix runs

Usage examples:
  # 1. Run baseline E0 on ISIC for 100 episodes on T4 GPU:
  modal run modal_runner.py --benchmark isic --experiment E0 --episodes 100

  # 2. Run proposed method E3 on DeepGlobe for 1000 episodes on A10G:
  modal run modal_runner.py --benchmark deepglobe --experiment E3 --episodes 1000 --gpu A10G

  # 3. Run all experiments E0-E3 on Lung for 100 episodes:
  modal run modal_runner.py --benchmark lung --experiment all --episodes 100
"""

import os
import sys
import json
import subprocess
import modal

# 1. Initialize Modal App and Persistent Volumes
app = modal.App("abcdfss-cd-fss")

volume_datasets = modal.Volume.from_name("abcdfss-datasets-cache", create_if_missing=True)
volume_results = modal.Volume.from_name("abcdfss-results", create_if_missing=True)

# 2. Define Container Image with CUDA dependencies
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "unzip", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "scipy",
        "tqdm",
        "kagglehub",
        "gdown",
    )
    .env({
        "KAGGLEHUB_CACHE": "/root/datasets_cache/kagglehub",
        "PYTHONPATH": "/root/ABCDFSS"
    })
    .add_local_dir(
        LOCAL_DIR,
        remote_path="/root/ABCDFSS",
        ignore=[".git", "*__pycache__*", "*.pdf", "*.log", "logs/*"]
    )
)

EXP_MAP = {
    'E0': {'adapter': 'conv1x1', 'fusion': 'mean'},
    'E1': {'adapter': 'depthwise_separable_3x3', 'fusion': 'mean'},
    'E2': {'adapter': 'conv1x1', 'fusion': 'softmax_margin'},
    'E3': {'adapter': 'depthwise_separable_3x3', 'fusion': 'softmax_margin'},
}

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={
        "/root/datasets_cache": volume_datasets,
        "/root/results": volume_results,
    }
)
def run_single_eval(
    benchmark: str = "deepglobe",
    experiment: str = "E0",
    nshot: int = 1,
    episodes: int = 1000,
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2
):
    import torch
    from run_all_benchmarks import check_or_download_dataset

    print("=" * 80)
    print("      ABCDFSS GPU EVALUATION ON MODAL CLOUD")
    print(f"Device:    {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Benchmark: {benchmark.upper()} | Shot: {nshot} | Seed: {seed}")
    print(f"Exp:       {experiment}")
    print(f"Mode:      {adapt_to} (Protocol: Every-Episode)")
    print(f"Episodes:  {episodes} episodes")
    print("=" * 80)

    # 1. Resolve dataset path (using persistent volume cache)
    kaggle_slugs = {
        'isic': 'heyoujue/isic2018-classwise',
        'suim': 'heyoujue/suim-merged',
        'lung': 'heyoujue/lungsegmentation',
        'deepglobe': 'heyoujue/deepglobe'
    }
    datapath = check_or_download_dataset(benchmark, kaggle_slug=kaggle_slugs.get(benchmark))
    volume_datasets.commit()

    # 2. Output directory inside persistent results volume
    output_dir = f"/root/results/{benchmark}/{experiment}_{episodes}ep_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Build evaluate.py execution command
    cmd = [
        "python", "evaluate.py",
        "--benchmark", benchmark,
        "--datapath", datapath,
        "--nshot", str(nshot),
        "--experiment", experiment,
        "--fusion-temp", str(fusion_temp),
        "--adapt-to", adapt_to,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--device", "cuda",
        "--nworker", str(nworker),
        "--logpath", output_dir
    ]

    if manifest:
        cmd.extend(["--manifest", manifest])

    print(f"\n[*] Executing: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ABCDFSS"

    process = subprocess.Popen(
        cmd,
        cwd="/root/ABCDFSS",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)

    process.wait()
    volume_results.commit()

    # Read and return result metrics
    result_json_path = os.path.join(output_dir, "run_result.json")
    metrics = {}
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
                metrics = res_data.get('metric', {})
        except Exception:
            pass

    print("=" * 80)
    print(f"[*] Completed {benchmark.upper()} {experiment} with exit code: {process.returncode}")
    if metrics:
        print(f"[*] Mean Episode-IoU: {metrics.get('Mean_Episode_IoU', 'N/A')}% | Cumulative mIoU: {metrics.get('Cumulative_mIoU', 'N/A')}%")
    print("=" * 80)

    return {
        "benchmark": benchmark,
        "experiment": experiment,
        "episodes": episodes,
        "seed": seed,
        "returncode": process.returncode,
        "metrics": metrics
    }

@app.local_entrypoint()
def main(
    benchmark: str = "deepglobe",
    experiment: str = "E0",
    nshot: int = 1,
    episodes: int = 1000,
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2
):
    benchmarks = ['deepglobe', 'isic', 'lung', 'fss', 'suim'] if benchmark == 'all' else [benchmark]
    experiments = ['E0', 'E1', 'E2', 'E3'] if experiment == 'all' else [experiment]

    all_results = []
    for b in benchmarks:
        for exp in experiments:
            print(f"\n>>> Dispatching Modal Job: Benchmark={b}, Exp={exp}, Episodes={episodes}")
            res = run_single_eval.remote(
                benchmark=b,
                experiment=exp,
                nshot=nshot,
                episodes=episodes,
                seed=seed,
                adapt_to=adapt_to,
                fusion_temp=fusion_temp,
                manifest=manifest,
                nworker=nworker
            )
            all_results.append(res)

    print("\n\n" + "=" * 80)
    print("              MODAL RUN SUMMARY RESULTS")
    print("=" * 80)
    for r in all_results:
        m = r.get('metrics', {})
        print(f"Benchmark: {r['benchmark']:<10} | Exp: {r['experiment']:<4} | Mean Ep IoU: {m.get('Mean_Episode_IoU', 'N/A')}% | Cum mIoU: {m.get('Cumulative_mIoU', 'N/A')}% | Code: {r['returncode']}")
    print("=" * 80)
