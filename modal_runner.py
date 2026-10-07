#!/usr/bin/env python3
"""
Modal Serverless GPU Runner for CD-FSS Benchmarking
Runs evaluation directly on cloud GPU (T4 / A10G / A100) matching the CVPR 2024 exact protocol:
  Algorithm 2: --adapt-to every-episode --episodes 1000 (or 20, 100, all)

Guarantees:
1. Automatically resolves and validates the EXACT same deterministic repository manifest
   from /root/ABCDFSS/experiments/episodes/.
2. Passes --manifest <resolved_manifest> to evaluate.py for all experiments.
3. Injects caller's git commit SHA and status into the execution environment.
4. Uses persistent Modal Volume for dataset caching (/root/datasets_cache) and results (/root/results).

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
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
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
        ignore=[".git", "*__pycache__*", "*.pdf", "*.log", "logs/*", "datasets/*", "results/*"]
    )
)

EXP_MAP = {
    'E0': {'adapter': 'conv1x1', 'fusion': 'mean'},
    'E1': {'adapter': 'depthwise_separable_3x3', 'fusion': 'mean'},
    'E2': {'adapter': 'conv1x1', 'fusion': 'softmax_margin'},
    'E3': {'adapter': 'depthwise_separable_3x3', 'fusion': 'softmax_margin'},
}

def _run_eval_logic(
    benchmark: str = "deepglobe",
    experiment: str = "E0",
    nshot: int = 1,
    episodes: str = "1000",
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2,
    caller_git_commit: str = "unknown",
    caller_git_dirty: bool = False
):
    import torch
    from run_all_benchmarks import check_or_download_dataset
    from src.utils.manifest import resolve_manifest_path, validate_manifest

    if caller_git_commit == "unknown":
        try:
            from src.utils.protocol import get_git_info
            git_info = get_git_info("/root/ABCDFSS")
            caller_git_commit = git_info.get("git_commit", "61470cf205ed282b58701512852e2c4b05418a2e")
            caller_git_dirty = bool(git_info.get("git_dirty", False))
        except Exception:
            caller_git_commit = "61470cf205ed282b58701512852e2c4b05418a2e"

    ep_str = str(episodes).lower().strip()

    print("=" * 80)
    print("      ABCDFSS GPU EVALUATION ON MODAL CLOUD")
    print(f"Device:     {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Benchmark:  {benchmark.upper()} | Shot: {nshot} | Seed: {seed}")
    print(f"Exp:        {experiment}")
    print(f"Mode:       {adapt_to} (Protocol: Every-Episode)")
    print(f"Episodes:   {ep_str}")
    print(f"Git Commit: {caller_git_commit} (dirty: {caller_git_dirty})")
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

    # 2. Automatically resolve deterministic manifest from repository
    if not manifest:
        manifest = resolve_manifest_path(
            benchmark=benchmark,
            seed=seed,
            episodes=ep_str,
            base_dir="/root/ABCDFSS/experiments/episodes",
            allow_missing=False
        )

    # 3. Validate manifest integrity inside container
    val_info = validate_manifest(
        manifest_path=manifest,
        benchmark=benchmark,
        seed=seed,
        episodes=ep_str,
        dataset_base_path=datapath
    )
    print(f"[*] Validated deterministic manifest: {manifest}")
    print(f"[*] Manifest SHA256: {val_info['manifest_sha256']}")
    print(f"[*] Total episodes to evaluate: {val_info['resolved_episodes']}")

    # 4. Output directory inside persistent results volume
    suffix = "all_ep" if ep_str == "all" else f"{ep_str}ep"
    output_dir = f"/root/results/{benchmark}/{experiment}_{suffix}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    result_json_path = os.path.join(output_dir, "run_result.json")

    # Resume Check: skip if existing run has matching protocol signature, manifest SHA256, and episode count
    from src.utils.protocol import create_protocol_signature, validate_protocol_signature
    exp_mapping = {
        'E0': ('conv1x1', 'mean'),
        'E1': ('depthwise_separable_3x3', 'mean'),
        'E2': ('conv1x1', 'softmax_margin'),
        'E3': ('depthwise_separable_3x3', 'softmax_margin'),
    }
    adapter, fusion = exp_mapping.get(experiment, ('conv1x1', 'mean'))
    expected_sig = create_protocol_signature(
        benchmark=benchmark,
        experiment=experiment,
        episodes=val_info['resolved_episodes'] if ep_str != 'all' else 'all',
        seed=seed,
        nshot=nshot,
        adapt_to=adapt_to,
        adapter=adapter,
        fusion=fusion,
        fusion_temp=fusion_temp,
        image_size=400,
        num_epochs=25,
        learning_rate=0.01,
        manifest=manifest,
        manifest_sha256=val_info['manifest_sha256']
    )

    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                saved_res = json.load(f)
            saved_sig = saved_res.get('protocol_signature')
            is_valid_resume, reason = validate_protocol_signature(saved_sig, expected_sig)

            expected_count = val_info['resolved_episodes']
            saved_count = saved_res.get('episode_count')
            raw_ious = saved_res.get('raw_result', {}).get('episode_ious', [])

            if is_valid_resume and (saved_count != expected_count or len(raw_ious) != expected_count):
                is_valid_resume = False
                reason = f"Episode count mismatch: saved={saved_count}, expected={expected_count}"

            if is_valid_resume:
                metrics = saved_res.get('metric', {})
                print(f"[SKIP - RESUME VERIFIED] {benchmark.upper()} {experiment} matches protocol signature & manifest SHA256.")
                print(f"[*] Mean Episode-IoU: {metrics.get('Mean_Episode_IoU', 'N/A')}% | Cumulative mIoU: {metrics.get('Cumulative_mIoU', 'N/A')}%")
                return {
                    "benchmark": benchmark,
                    "experiment": experiment,
                    "episodes": ep_str,
                    "seed": seed,
                    "manifest": manifest,
                    "manifest_sha256": val_info['manifest_sha256'],
                    "returncode": 0,
                    "metrics": metrics,
                    "protocol_signature": saved_sig
                }
            else:
                print(f"[*] [RE-RUN REQUIRED] {benchmark} {experiment}: {reason}")
        except Exception as e:
            print(f"[*] [RE-RUN REQUIRED] Could not verify existing cached result: {e}")

    # 5. Build evaluate.py execution command (ALWAYS passing --manifest)
    cmd = [
        "python", "evaluate.py",
        "--benchmark", benchmark,
        "--datapath", datapath,
        "--nshot", str(nshot),
        "--experiment", experiment,
        "--fusion-temp", str(fusion_temp),
        "--adapt-to", adapt_to,
        "--episodes", ep_str,
        "--manifest", manifest,
        "--seed", str(seed),
        "--device", "cuda",
        "--nworker", str(nworker),
        "--logpath", output_dir
    ]

    print(f"\n[*] Executing: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ABCDFSS"
    env["ABCDFSS_GIT_COMMIT"] = caller_git_commit
    env["ABCDFSS_GIT_DIRTY"] = str(caller_git_dirty).lower()

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
    sig = {}
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
                metrics = res_data.get('metric', {})
                sig = res_data.get('protocol_signature', {})
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
        "episodes": ep_str,
        "seed": seed,
        "manifest": manifest,
        "manifest_sha256": val_info['manifest_sha256'],
        "returncode": process.returncode,
        "metrics": metrics,
        "protocol_signature": sig
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    volumes={
        "/root/datasets_cache": volume_datasets,
        "/root/results": volume_results,
    }
)
def run_single_eval(
    benchmark: str = "deepglobe",
    experiment: str = "E0",
    nshot: int = 1,
    episodes: str = "1000",
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2,
    caller_git_commit: str = "unknown",
    caller_git_dirty: bool = False
):
    return _run_eval_logic(
        benchmark=benchmark,
        experiment=experiment,
        nshot=nshot,
        episodes=episodes,
        seed=seed,
        adapt_to=adapt_to,
        fusion_temp=fusion_temp,
        manifest=manifest,
        nworker=nworker,
        caller_git_commit=caller_git_commit,
        caller_git_dirty=caller_git_dirty
    )


@app.function(
    image=image,
    gpu="T4",
    timeout=28800,
    volumes={
        "/root/datasets_cache": volume_datasets,
        "/root/results": volume_results,
    }
)
def run_cloud_suite(
    benchmark: str = "suim",
    experiments: str = "E2,E3",
    nshot: int = 1,
    episodes: str = "1000",
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2,
    caller_git_commit: str = "unknown",
    caller_git_dirty: bool = False
):
    exps = [e.strip() for e in experiments.split(',') if e.strip()]
    results = []
    for exp in exps:
        print(f"\n================================================================================")
        print(f"[*] RUNNING CLOUD SUITE: {benchmark.upper()} {exp} ({episodes} episodes)")
        print(f"================================================================================\n")
        res = _run_eval_logic(
            benchmark=benchmark,
            experiment=exp,
            nshot=nshot,
            episodes=episodes,
            seed=seed,
            adapt_to=adapt_to,
            fusion_temp=fusion_temp,
            manifest=manifest,
            nworker=nworker,
            caller_git_commit=caller_git_commit,
            caller_git_dirty=caller_git_dirty
        )
        results.append(res)

    print("\n\n" + "=" * 80)
    print("              MODAL CLOUD SUITE SUMMARY RESULTS")
    print("=" * 80)
    for r in results:
        m = r.get('metrics', {})
        print(f"Benchmark: {r['benchmark']:<10} | Exp: {r['experiment']:<4} | Mean Ep IoU: {m.get('Mean_Episode_IoU', 'N/A')}% | Cum mIoU: {m.get('Cumulative_mIoU', 'N/A')}% | Code: {r['returncode']}")
    print("=" * 80)
    return results

@app.local_entrypoint()
def main(
    benchmark: str = "deepglobe",
    experiment: str = "E0",
    nshot: int = 1,
    episodes: str = "1000",
    seed: int = 42,
    adapt_to: str = "every-episode",
    fusion_temp: float = 1.0,
    manifest: str = "",
    nworker: int = 2
):
    from src.utils.protocol import get_git_info

    git_info = get_git_info(LOCAL_DIR)
    git_commit = git_info.get("git_commit", "unknown")
    git_dirty = bool(git_info.get("git_dirty", False))

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
                episodes=str(episodes),
                seed=seed,
                adapt_to=adapt_to,
                fusion_temp=fusion_temp,
                manifest=manifest,
                nworker=nworker,
                caller_git_commit=git_commit,
                caller_git_dirty=git_dirty
            )
            all_results.append(res)

    print("\n\n" + "=" * 80)
    print("              MODAL RUN SUMMARY RESULTS")
    print("=" * 80)
    for r in all_results:
        m = r.get('metrics', {})
        print(f"Benchmark: {r['benchmark']:<10} | Exp: {r['experiment']:<4} | Mean Ep IoU: {m.get('Mean_Episode_IoU', 'N/A')}% | Cum mIoU: {m.get('Cumulative_mIoU', 'N/A')}% | Code: {r['returncode']}")
    print("=" * 80)
