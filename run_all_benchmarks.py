#!/usr/bin/env python3
"""
Automated Master Runner for All 5 CD-FSS Benchmarks
Runs full evaluation across all domains, saves detailed individual logs,
and outputs a comprehensive verified benchmark summary table.

Key Guarantees:
1. Automatically resolves and validates explicit deterministic manifests.
2. The exact same manifest is shared across E0, E1, E2, and E3.
3. Cryptographic protocol signature validation prevents invalid resume skips.
4. Outputs results directly into results/{dataset}/{exp}_{episodes}ep_seed{seed}/.

Usage:
    python run_all_benchmarks.py
    python run_all_benchmarks.py --experiments E0 E1 E2 E3 --episodes 100 --device cuda
    python run_all_benchmarks.py --benchmarks deepglobe isic lung fss suim --episodes 100
"""

import os
import sys
import json
import time
import argparse
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.manifest import resolve_manifest_path, validate_manifest
from src.utils.protocol import create_protocol_signature, validate_protocol_signature


def is_valid_fss_dir(path: str) -> bool:
    """
    Checks whether `path` DIRECTLY contains FSS class folders (with images/masks).
    Does NOT recurse arbitrarily to avoid false-positives on parent datasets directories.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    try:
        from src.datasets.fss import FSS_TEST_CLASSES
        entries = set(os.listdir(path))
        matching_classes = entries.intersection(set(FSS_TEST_CLASSES))
        if len(matching_classes) >= 5:
            sample_cls = next(iter(matching_classes))
            cls_p = os.path.join(path, sample_cls)
            if os.path.isdir(cls_p) and any(f.lower().endswith(('.jpg', '.png')) for f in os.listdir(cls_p)):
                return True
    except Exception:
        pass
    return False


def find_fss_root(start_dir: str) -> Optional[str]:
    """Recursively searches start_dir to find the directory that directly contains FSS class folders."""
    if not os.path.exists(start_dir):
        return None
    if is_valid_fss_dir(start_dir):
        return os.path.abspath(start_dir)
    for root, dirs, files in os.walk(start_dir):
        if is_valid_fss_dir(root):
            return os.path.abspath(root)
    return None


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
        import zipfile
        from tqdm import tqdm

        fss_check_dirs = [
            "/root/datasets_cache/fewshot1000",
            "/root/datasets_cache/fss1000",
            "./datasets/fewshot1000",
            "./datasets/fss1000",
            "./datasets/fss",
            os.path.join(PROJECT_ROOT, "datasets", "fss1000"),
            os.path.join(PROJECT_ROOT, "datasets", "fewshot1000"),
        ]
        for cd in fss_check_dirs:
            valid_root = find_fss_root(cd)
            if valid_root:
                print(f"[OK] Found valid FSS-1000 at: {valid_root}")
                return valid_root

        # Determine target cache directory (Modal persistent volume cache if available, else local datasets)
        cache_base = "/root/datasets_cache" if os.path.exists("/root/datasets_cache") else "./datasets"
        target_dir = os.path.join(cache_base, "fewshot1000")
        os.makedirs(target_dir, exist_ok=True)
        zip_path = os.path.join(target_dir, "fewshot1000.zip")

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
                print(f"[*] Extracting FSS-1000 into {target_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                valid_root = find_fss_root(target_dir)
                if valid_root:
                    print(f"[OK] Downloaded and extracted FSS-1000 to: {valid_root}")
                    return valid_root
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
                    print(f"[*] Extracting FSS-1000 into {target_dir}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                    valid_root = find_fss_root(target_dir)
                    if valid_root:
                        print(f"[OK] Downloaded and extracted FSS-1000 to: {valid_root}")
                        return valid_root
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
    parser = argparse.ArgumentParser(description="Automated Master Runner for All 5 CD-FSS Benchmarks")
    parser.add_argument('--adapt-to', type=str, default='every-episode',
                        choices=['first-episode', 'every-episode'],
                        help='Adaptation mode: every-episode (CVPR 2024 exact protocol) or first-episode')
    parser.add_argument('--episodes', type=str, default='100',
                        help='Number of episodes per benchmark: 20, 100, 1000, or "all" (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic evaluation (default: 42)')
    parser.add_argument('--experiments', nargs='+', default=['E0', 'E1', 'E2', 'E3'],
                        choices=['E0', 'E1', 'E2', 'E3', 'all'],
                        help="Experiments to run: E0, E1, E2, E3, or all (default: E0 E1 E2 E3)")
    parser.add_argument('--benchmarks', nargs='+', default=['deepglobe', 'isic', 'lung', 'fss', 'suim'],
                        help='List of benchmarks to run (default: deepglobe isic lung fss suim)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device: cuda or cpu (default: cuda)')
    parser.add_argument('--nshot', type=int, default=1,
                        help='Few-shot k value (default: 1)')
    parser.add_argument('--nworker', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (default: results/{benchmark}/{exp}_{episodes}ep_seed{seed})')
    return parser.parse_args()


def main():
    args = parse_args()
    ep_str = str(args.episodes).lower().strip()

    print("=" * 80)
    print("      AUTOMATED MASTER CD-FSS BENCHMARK PIPELINE")
    print(f"      Mode:      {args.adapt_to} (Protocol: CVPR 2024)")
    print(f"      Episodes:  {ep_str} | Shot: {args.nshot} | Seed: {args.seed} | Device: {args.device}")
    print(f"      Exps:      {args.experiments}")
    print(f"      Domains:   {args.benchmarks}")
    print("=" * 80 + "\n")

    # 1. Resolve dataset paths
    all_slugs = {
        'isic': 'heyoujue/isic2018-classwise',
        'suim': 'heyoujue/suim-merged',
        'lung': 'heyoujue/lungsegmentation',
        'fss': None,
        'fss1000': None,
        'deepglobe': 'heyoujue/deepglobe'
    }

    dataset_paths = {}
    for b in args.benchmarks:
        b_key = b.lower().replace('-1000', '').replace('1000', '')
        if b_key in all_slugs:
            dataset_paths[b] = check_or_download_dataset(b_key, kaggle_slug=all_slugs.get(b_key))

    # 2. Experiment definition matrix
    exp_definitions = {
        'E0': {'name': 'E0: Baseline (Conv1x1 + Mean)', 'adapter': 'conv1x1', 'fusion': 'mean'},
        'E1': {'name': 'E1: Adapter Ablation (DW3x3 + Mean)', 'adapter': 'depthwise_separable_3x3', 'fusion': 'mean'},
        'E2': {'name': 'E2: Fusion Ablation (Conv1x1 + Softmax)', 'adapter': 'conv1x1', 'fusion': 'softmax_margin'},
        'E3': {'name': 'E3: Full Proposed (DW3x3 + Softmax)', 'adapter': 'depthwise_separable_3x3', 'fusion': 'softmax_margin'},
    }
    selected_exp_keys = ['E0', 'E1', 'E2', 'E3'] if 'all' in args.experiments else args.experiments

    # 3. Resolve and validate deterministic manifests per benchmark
    manifest_info = {}
    for b in args.benchmarks:
        b_norm = b.lower().replace('-1000', '').replace('1000', '')
        if b not in dataset_paths or not dataset_paths[b]:
            print(f"[!] Skipping {b}: dataset not found.")
            continue
        try:
            m_path = resolve_manifest_path(
                benchmark=b_norm,
                seed=args.seed,
                episodes=ep_str,
                allow_missing=False
            )
            v_info = validate_manifest(
                manifest_path=m_path,
                benchmark=b_norm,
                seed=args.seed,
                episodes=ep_str,
                dataset_base_path=dataset_paths[b]
            )
            manifest_info[b] = {
                'benchmark': b_norm,
                'path': m_path,
                'sha256': v_info['manifest_sha256'],
                'resolved_episodes': v_info['resolved_episodes']
            }
            print(f"[OK] Manifest locked for {b.upper()}: {os.path.basename(m_path)} (SHA: {v_info['manifest_sha256'][:10]}...)")
        except Exception as me:
            print(f"[!] Warning: Could not resolve/validate manifest for {b}: {me}")

    # 4. Execute experiments sequentially
    total_runs = len(manifest_info) * len(selected_exp_keys)
    current_run = 0
    records = []

    for b, m_data in manifest_info.items():
        b_norm = m_data['benchmark']
        datapath = dataset_paths[b]
        manifest_path = m_data['path']
        manifest_sha = m_data['sha256']
        resolved_ep = m_data['resolved_episodes']

        for ek in selected_exp_keys:
            current_run += 1
            ed = exp_definitions[ek]

            # Output directory
            if args.output_dir:
                run_dir = os.path.join(args.output_dir, b, ek)
            else:
                suffix = "all_ep" if ep_str == "all" else f"{ep_str}ep"
                run_dir = os.path.join(PROJECT_ROOT, "results", b_norm, f"{ek}_{suffix}_seed{args.seed}")
            os.makedirs(run_dir, exist_ok=True)
            result_json = os.path.join(run_dir, "run_result.json")

            # Check Protocol Signature for resume
            expected_sig = create_protocol_signature(
                benchmark=b_norm,
                experiment=ek,
                episodes=resolved_ep if ep_str != 'all' else 'all',
                seed=args.seed,
                nshot=args.nshot,
                adapt_to=args.adapt_to,
                adapter=ed['adapter'],
                fusion=ed['fusion'],
                fusion_temp=1.0,
                image_size=400,
                num_epochs=25,
                learning_rate=0.01,
                manifest=manifest_path,
                manifest_sha256=manifest_sha
            )

            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        saved_res = json.load(f)
                    saved_sig = saved_res.get('protocol_signature')
                    is_valid, reason = validate_protocol_signature(saved_sig, expected_sig)

                    saved_count = saved_res.get('episode_count')
                    if is_valid and saved_count == resolved_ep:
                        m = saved_res.get('metric', {})
                        print(f"[{current_run:02d}/{total_runs:02d}] [SKIP - RESUME VERIFIED] {b.upper()} {ek} => Mean Ep IoU: {m.get('Mean_Episode_IoU', 'N/A')}%, Cum mIoU: {m.get('Cumulative_mIoU', 'N/A')}%")
                        records.append({
                            'benchmark': b.upper(),
                            'experiment': ek,
                            'adapter': ed['adapter'],
                            'fusion': ed['fusion'],
                            'Mean_Episode_IoU': m.get('Mean_Episode_IoU', 0.0),
                            'Cumulative_mIoU': m.get('Cumulative_mIoU', 0.0),
                            'FB-IoU': m.get('FB-IoU', 0.0),
                            'log_file': saved_res.get('log', '')
                        })
                        continue
                    else:
                        print(f"[{current_run:02d}/{total_runs:02d}] [RE-RUN] {b.upper()} {ek}: {reason}")
                except Exception as ex:
                    print(f"[{current_run:02d}/{total_runs:02d}] [RE-RUN] {b.upper()} {ek}: {ex}")

            print(f"\n{'='*70}")
            print(f"[{current_run:02d}/{total_runs:02d}] RUNNING: {b.upper()} - {ed['name']}")
            print(f"Manifest: {os.path.basename(manifest_path)}")
            print(f"Output:   {run_dir}")
            print(f"{'='*70}\n")

            cmd = [
                sys.executable, os.path.join(PROJECT_ROOT, "evaluate.py"),
                "--benchmark", b_norm,
                "--datapath", datapath,
                "--nshot", str(args.nshot),
                "--experiment", ek,
                "--adapter", ed['adapter'],
                "--fusion", ed['fusion'],
                "--adapt-to", args.adapt_to,
                "--episodes", ep_str,
                "--manifest", manifest_path,
                "--seed", str(args.seed),
                "--device", args.device,
                "--nworker", str(args.nworker),
                "--logpath", run_dir
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = PROJECT_ROOT + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

            t0 = time.time()
            res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            elapsed = time.time() - t0

            if res.returncode == 0:
                print(f"[OK] Completed {b.upper()} {ek} in {elapsed:.1f}s")
                if os.path.exists(result_json):
                    try:
                        with open(result_json, 'r', encoding='utf-8') as f:
                            saved_res = json.load(f)
                        m = saved_res.get('metric', {})
                        records.append({
                            'benchmark': b.upper(),
                            'experiment': ek,
                            'adapter': ed['adapter'],
                            'fusion': ed['fusion'],
                            'Mean_Episode_IoU': m.get('Mean_Episode_IoU', 0.0),
                            'Cumulative_mIoU': m.get('Cumulative_mIoU', 0.0),
                            'FB-IoU': m.get('FB-IoU', 0.0),
                            'log_file': saved_res.get('log', '')
                        })
                    except Exception:
                        pass
            else:
                print(f"[!] ERROR in {b.upper()} {ek} (exit code: {res.returncode})")

    # 5. Output Summary Table
    if records:
        print("\n\n" + "=" * 80)
        print("           VERIFIED EXPERIMENT BENCHMARK SUMMARY TABLE")
        print("=" * 80)
        md_table = [
            "| Domain | Exp | Adapter | Fusion | Mean Ep IoU (%) | Cum mIoU (%) | FB-IoU (%) |",
            "| :--- | :---: | :--- | :--- | :---: | :---: | :---: |"
        ]
        for r in records:
            md_table.append(f"| **{r['benchmark']}** | `{r['experiment']}` | `{r['adapter']}` | `{r['fusion']}` | **{r['Mean_Episode_IoU']:.2f}%** | **{r['Cumulative_mIoU']:.2f}%** | **{r['FB-IoU']:.2f}%** |")

        table_str = "\n".join(md_table)
        print(table_str)
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
