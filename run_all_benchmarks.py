#!/usr/bin/env python3
"""
Automated Master Runner for All 5 CD-FSS Benchmarks
Runs full evaluation across all domains, saves detailed individual logs,
and outputs a comprehensive verified benchmark summary table.

Usage:
    python run_all_benchmarks.py
    python run_all_benchmarks.py --experiments E0 E1 E2 E3 --episodes 100 --device cuda
"""

import os
import sys
import json
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def is_valid_fss_dir(path: str) -> bool:
    """Checks whether the FSS directory contains at least 10 classes and valid images."""
    if not os.path.exists(path):
        return False
    valid_classes = 0
    for root, dirs, files in os.walk(path):
        jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
        if len(jpgs) >= 2:
            valid_classes += 1
            if valid_classes >= 10:
                return True
    return False

def check_or_download_dataset(benchmark: str, kaggle_slug: str = None) -> str:
    """
    Robust dataset path locator and downloader.
    Checks local directories and Kaggle cache roots across Windows, Linux, and Modal.
    Downloads automatically if not found.
    """
    # 1. Check local project directory first
    local_candidates = [
        f"./datasets/{benchmark}",
        f"./datasets/{benchmark}1000",
        os.path.join(PROJECT_ROOT, "datasets", benchmark),
        os.path.join(PROJECT_ROOT, "datasets", f"{benchmark}1000")
    ]
    for lc in local_candidates:
        if os.path.exists(lc) and os.path.isdir(lc) and len(os.listdir(lc)) > 0:
            print(f"[OK] Found '{benchmark}' in local directory: {lc}")
            return lc

    # 2. Check Kaggle cache roots (Windows, Linux, Modal persistent volume)
    cache_roots = []
    if os.environ.get("KAGGLEHUB_CACHE"):
        cache_roots.append(os.environ.get("KAGGLEHUB_CACHE"))
    cache_roots.extend([
        os.path.expanduser("~/.cache/kagglehub"),
        "/root/.cache/kagglehub",
        "/root/datasets_cache/kagglehub",
        "/root/datasets_cache"
    ])

    if kaggle_slug:
        slug_parts = kaggle_slug.split("/")
        for cr in cache_roots:
            slug_dir = os.path.join(cr, "datasets", *slug_parts, "versions")
            if os.path.exists(slug_dir) and os.path.isdir(slug_dir):
                versions = [v for v in os.listdir(slug_dir) if os.path.isdir(os.path.join(slug_dir, v))]
                if versions:
                    try:
                        versions.sort(key=lambda x: int(x), reverse=True)
                    except ValueError:
                        versions.sort(reverse=True)
                    chosen = os.path.join(slug_dir, versions[0])
                    print(f"[OK] Found cached '{benchmark}' in: {chosen}")
                    return chosen

    # 3. Special handling for FSS-1000
    if benchmark in ['fss', 'fss1000']:
        import shutil, zipfile, urllib.request
        from tqdm import tqdm

        fss_check_dirs = [
            "./datasets/fewshot1000",
            "./datasets/fss1000",
            "./datasets/fss",
            os.path.join(PROJECT_ROOT, "datasets", "fss1000"),
            os.path.join(PROJECT_ROOT, "datasets", "fewshot1000"),
            "/root/datasets_cache/fewshot1000",
            "/root/datasets_cache/fss1000"
        ]
        for cd in fss_check_dirs:
            if is_valid_fss_dir(cd):
                for root, dirs, files in os.walk(cd):
                    if len(dirs) >= 10:
                        sample_sub = dirs[0]
                        sub_p = os.path.join(root, sample_sub)
                        if os.path.isdir(sub_p):
                            try:
                                if any(f.endswith('.jpg') for f in os.listdir(sub_p)):
                                    print(f"[OK] Found valid FSS-1000 at: {root}")
                                    return root
                            except Exception:
                                pass
                print(f"[OK] Found valid FSS-1000 at: {cd}")
                return cd

        # Clean corrupted directories before fresh download
        for bad_dir in ["./datasets/fss1000", "./datasets/fewshot1000"]:
            if os.path.exists(bad_dir):
                shutil.rmtree(bad_dir, ignore_errors=True)

        os.makedirs("./datasets", exist_ok=True)
        zip_path = "./datasets/fss1000.zip"

        # Priority 1: Hugging Face CDN (fast, unauthenticated)
        hf_url = "https://huggingface.co/datasets/zhaoyuan666/ConceptSeg-Benchmark/resolve/main/fewshot1000.zip"
        try:
            print("[*] Downloading FSS-1000 from Hugging Face CDN (679 MB)...")
            req = urllib.request.Request(hf_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out_f:
                total_size = int(resp.headers.get('Content-Length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="fewshot1000.zip") as pbar:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        pbar.update(len(chunk))

            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000000:
                print("[*] Extracting FSS-1000...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("./datasets")
                for root, dirs, files in os.walk("./datasets"):
                    if is_valid_fss_dir(root):
                        print(f"[OK] Downloaded and extracted FSS-1000 to: {root}")
                        return root
        except Exception as hf_err:
            print(f"[-] Hugging Face download failed ({hf_err}), falling back to Google Drive...")

        # Priority 2: Google Drive via gdown
        gdrive_ids = ["16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI", "1tt3dkdASjXt58t-2A9zeucZ397ZRF7In"]
        for gid in gdrive_ids:
            try:
                import gdown
                if os.path.exists(zip_path) and os.path.getsize(zip_path) < 10000000:
                    os.remove(zip_path)
                if not os.path.exists(zip_path):
                    print(f"[*] Downloading fss1000.zip from Google Drive ID ({gid})...")
                    gdown.download(id=gid, output=zip_path, quiet=False)
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 10000000:
                    print("[*] Extracting FSS-1000...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall("./datasets/fss1000")
                    for root, dirs, files in os.walk("./datasets/fss1000"):
                        if is_valid_fss_dir(root):
                            print(f"[OK] Downloaded and extracted FSS-1000 to: {root}")
                            return root
            except Exception as gde:
                print(f"[-] Failed with Google Drive ID {gid}: {gde}")

        print("[!] Unable to download FSS-1000 automatically.")
        return "./datasets/fss1000"

    # 4. Download from Kaggle via kagglehub
    if kaggle_slug:
        print(f"[*] Downloading '{benchmark}' from Kaggle ({kaggle_slug})...")
        try:
            import kagglehub
            path = kagglehub.dataset_download(kaggle_slug)
            print(f"[OK] Successfully downloaded '{benchmark}' to: {path}")
            return path
        except Exception as e:
            print(f"[!] Error downloading {benchmark} from kagglehub: {e}")

    return f"./datasets/{benchmark}"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Automated Runner for CD-FSS Benchmarks")
    parser.add_argument('--adapt-to', type=str, default='every-episode',
                        choices=['first-episode', 'every-episode'],
                        help='Adaptation mode: every-episode (CVPR 2024 exact protocol) or first-episode (quick-infer)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes per benchmark (default: 1000 standard CVPR episodes)')
    parser.add_argument('--experiments', nargs='+', default=['E0'],
                        choices=['E0', 'E1', 'E2', 'E3', 'all'],
                        help="Experiments to run: E0 (baseline: conv1x1+mean), E1 (dw3x3+mean), E2 (conv1x1+softmax), E3 (dw3x3+softmax), or all (default: E0)")
    parser.add_argument('--benchmarks', nargs='+', default=['isic', 'suim', 'lung', 'fss', 'deepglobe'],
                        help='List of benchmarks to run')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device: cuda or cpu')
    parser.add_argument('--nworker', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 80)
    print(f"      CD-FSS BENCHMARK PIPELINE: {args.adapt_to.upper()} ({args.episodes} EPISODES)")
    print(f"      EXPERIMENTS: {args.experiments}")
    print("=" * 80)

    # 1. Resolve dataset paths
    all_slugs = {
        'isic': 'heyoujue/isic2018-classwise',
        'suim': 'heyoujue/suim-merged',
        'lung': 'heyoujue/lungsegmentation',
        'fss': None,
        'deepglobe': 'heyoujue/deepglobe'
    }
    dataset_paths = {}
    for b in args.benchmarks:
        if b in all_slugs:
            dataset_paths[b] = check_or_download_dataset(b, kaggle_slug=all_slugs[b])

    # 2. Build 4 standard experiment matrix (E0/E1/E2/E3)
    exp_definitions = {
        'E0': {'name': 'E0 — Original ABCDFSS Baseline', 'adapter': 'conv1x1', 'fusion': 'mean'},
        'E1': {'name': 'E1 — Adapter Ablation (DW3x3 + Mean)', 'adapter': 'depthwise_separable_3x3', 'fusion': 'mean'},
        'E2': {'name': 'E2 — Fusion Ablation (Conv1x1 + Softmax)', 'adapter': 'conv1x1', 'fusion': 'softmax_margin'},
        'E3': {'name': 'E3 — Full Proposed Method (DW3x3 + Softmax)', 'adapter': 'depthwise_separable_3x3', 'fusion': 'softmax_margin'},
    }

    selected_exp_keys = ['E0', 'E1', 'E2', 'E3'] if 'all' in args.experiments else args.experiments

    experiments = []
    for b in args.benchmarks:
        if b not in dataset_paths or not dataset_paths[b]:
            continue
        for ek in selected_exp_keys:
            ed = exp_definitions[ek]
            experiments.append({
                'id': ek,
                'name': f"{b.upper()} [{ed['name']}]",
                'benchmark': b,
                'datapath': dataset_paths[b],
                'adapter': ed['adapter'],
                'fusion': ed['fusion'],
                'nshot': 1
            })

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 3. Execute experiments sequentially
    total_exp = len(experiments)
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*30} PROGRESS [{i}/{total_exp}]: {exp['name']} {'='*30}")
        if not os.path.exists(exp['datapath']) or (exp['benchmark'] == 'fss' and not is_valid_fss_dir(exp['datapath'])):
            print(f"[!] WARNING: Dataset directory '{exp['benchmark']}' at {exp['datapath']} is invalid or missing. Skipping...")
            continue

        cmd = [
            sys.executable, "evaluate.py",
            "--benchmark", exp['benchmark'],
            "--datapath", exp['datapath'],
            "--nshot", str(exp['nshot']),
            "--experiment", exp['id'],
            "--adapter", exp['adapter'],
            "--fusion", exp['fusion'],
            "--adapt-to", args.adapt_to,
            "--episodes", str(args.episodes),
            "--logpath", log_dir,
            "--device", args.device,
            "--nworker", str(args.nworker)
        ]

        print(f"[*] Executing command: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            print(f"[!] Error executing experiment {exp['name']}")

    # 4. Read summary_records.jsonl and output summary table
    summary_jsonl = os.path.join(log_dir, "summary_records.jsonl")
    if os.path.exists(summary_jsonl):
        records = []
        with open(summary_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        print("\n\n" + "=" * 80)
        print("           VERIFIED EXPERIMENT BENCHMARK SUMMARY TABLE")
        print("=" * 80)
        md_table = [
            "| Timestamp | Benchmark | Shot | Adapter | Fusion | Mean Ep IoU (%) | Cum mIoU (%) | FB-IoU (%) | Log File |",
            "| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |"
        ]
        for r in records:
            mean_ep = r.get('Mean_Episode_IoU', r.get('mIoU', 'N/A'))
            cum_m = r.get('Cumulative_mIoU', r.get('mIoU', 'N/A'))
            md_table.append(f"| {r['timestamp']} | **{r['benchmark'].upper()}** | {r['nshot']} | `{r['adapter']}` | `{r['fusion']}` | **{mean_ep}%** | **{cum_m}%** | **{r['FB-IoU']}%** | `{os.path.basename(r['log_file'])}` |")

        table_str = "\n".join(md_table)
        print(table_str)

        output_md_file = os.path.join(log_dir, "verified_benchmark_summary.md")
        with open(output_md_file, "w", encoding="utf-8") as f:
            f.write("# VERIFIED BENCHMARK EXPERIMENT SUMMARY\n\n" + table_str + "\n")
        print(f"\n[OK] Summary exported to: {output_md_file}")

if __name__ == '__main__':
    main()
