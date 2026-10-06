#!/usr/bin/env python3
"""
Cross-Domain Few-Shot Segmentation (CD-FSS) Evaluation
Independent, clean self-built framework.

Usage examples:
    python evaluate.py --benchmark isic --datapath /path/to/isic --experiment E0
    python evaluate.py --benchmark deepglobe --datapath /path/to/deepglobe --experiment E3 --episodes 100
    python evaluate.py --benchmark lung --datapath /path/to/lung --experiment E0 --episodes all
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
import random
import numpy as np
import torchvision
from datetime import datetime
import json

from src.datasets import build_dataloader
from src.engine import CDFSSEngine
from src.utils.manifest import resolve_manifest_path, validate_manifest, compute_manifest_sha256
from src.utils.protocol import get_git_info, get_environment_info, create_protocol_signature

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Clean CD-FSS Framework Evaluation")
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['fss', 'fss1000', 'isic', 'lung', 'chest', 'deepglobe', 'suim'],
                        help='Dataset benchmark name')
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--nshot', type=int, default=1,
                        help='Number of support shots (default: 1)')
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['E0', 'E1', 'E2', 'E3'],
                        help='Standard experiment shortcut: E0 (baseline: conv1x1+mean), E1 (adapter: dw3x3+mean), E2 (fusion: conv1x1+softmax), E3 (proposed: dw3x3+softmax)')
    parser.add_argument('--adapter', '--adapter-type', dest='adapter', type=str, default='conv1x1',
                        choices=['depthwise_separable_3x3', 'conv1x1', 'pointwise'],
                        help='Adapter architecture: conv1x1 (baseline) or depthwise_separable_3x3 (proposed)')
    parser.add_argument('--fusion', '--fusion-mode', dest='fusion', type=str, default='mean',
                        choices=['softmax_margin', 'mean', 'learnable'],
                        help='Multi-layer fusion method: mean (baseline) or softmax_margin (proposed)')
    parser.add_argument('--fusion-temp', type=float, default=1.0,
                        help='Softmax temperature scaling factor')
    parser.add_argument('--adapt-to', type=str, default='every-episode',
                        choices=['first-episode', 'every-episode'],
                        help='Adaptation mode: every-episode (standard CVPR Algorithm 2) or first-episode (quick-infer)')
    parser.add_argument('--episodes', type=str, default='20',
                        help='Number of episodes to evaluate (e.g. 20, 100, 1000, or "all")')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to pre-generated episode manifest JSON')
    parser.add_argument('--allow-runtime-sampling', action='store_true',
                        help='Allow ad-hoc random sampling without manifest (disabled by default in benchmark mode)')
    parser.add_argument('--postprocessing', type=str, default='off',
                        help='Postprocessing option (off by default)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--logpath', type=str, default='./logs',
                        help='Directory to save evaluation log files')
    parser.add_argument('--img-size', type=int, default=400,
                        help='Input image resolution (default: 400)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--nworker', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    return parser.parse_args()

class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def main():
    args = parse_args()

    # Resolve explicit standard experiment configurations
    if args.experiment == 'E0':
        args.adapter = 'conv1x1'
        args.fusion = 'mean'
    elif args.experiment == 'E1':
        args.adapter = 'depthwise_separable_3x3'
        args.fusion = 'mean'
    elif args.experiment == 'E2':
        args.adapter = 'conv1x1'
        args.fusion = 'softmax_margin'
    elif args.experiment == 'E3':
        args.adapter = 'depthwise_separable_3x3'
        args.fusion = 'softmax_margin'

    if args.adapter == 'pointwise':
        args.adapter = 'conv1x1'

    # 1. Deterministic Manifest Resolution & Validation
    manifest_sha256 = None
    manifest_info = None

    if not args.manifest and not args.allow_runtime_sampling:
        args.manifest = resolve_manifest_path(
            benchmark=args.benchmark,
            seed=args.seed,
            episodes=args.episodes,
            manifest_path=None,
            allow_missing=False
        )

    if args.manifest:
        manifest_info = validate_manifest(
            manifest_path=args.manifest,
            benchmark=args.benchmark,
            seed=args.seed,
            episodes=args.episodes,
            dataset_base_path=args.datapath
        )
        args.manifest = manifest_info["manifest_path"]
        manifest_sha256 = manifest_info["manifest_sha256"]
        resolved_episodes = manifest_info["resolved_episodes"]
    else:
        # Fallback to runtime sampling only if explicitly allowed
        resolved_episodes = None if str(args.episodes).lower() == 'all' else int(args.episodes)

    set_seed(args.seed)

    # Setup automatic file logging
    os.makedirs(args.logpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment if args.experiment else f"{args.adapter}_{args.fusion}"
    exp_tag = f"_{args.experiment}" if args.experiment else ""
    log_filename = f"{args.benchmark}_{args.adapter}_{args.fusion}_{args.nshot}shot{exp_tag}_{timestamp}.log"
    log_filepath = os.path.join(args.logpath, log_filename)
    logger = TeeLogger(log_filepath)
    sys.stdout = logger

    env_info = get_environment_info()
    git_info = get_git_info()

    protocol_sig = create_protocol_signature(
        benchmark=args.benchmark,
        experiment=exp_name,
        episodes=resolved_episodes,
        seed=args.seed,
        nshot=args.nshot,
        adapt_to=args.adapt_to,
        adapter=args.adapter,
        fusion=args.fusion,
        fusion_temp=args.fusion_temp,
        image_size=args.img_size,
        num_epochs=25,
        learning_rate=1e-2,
        manifest=args.manifest,
        manifest_sha256=manifest_sha256
    )

    print("=" * 80)
    print("                     CD-FSS EXPERIMENTAL AUDIT RUNNER")
    print("=" * 80)
    print(f"experiment_name:    {exp_name}")
    print(f"dataset:            {args.benchmark}")
    print(f"seed:               {args.seed}")
    print(f"nshot:              {args.nshot}")
    print(f"requested_episodes: {args.episodes}")
    print(f"resolved_episodes:  {resolved_episodes}")
    print(f"adapter:            {args.adapter}")
    print(f"fusion:             {args.fusion}")
    print(f"adaptation_mode:    {args.adapt_to}")
    print(f"image_size:         {args.img_size}x{args.img_size}")
    print(f"threshold_method:   pred_mean (max(Otsu, mean), drop_least=0.05)")
    print(f"backbone:           ResNet-50 (Pre-ReLU unclipped features)")
    print(f"checkpoint:         ResNet50_Weights.DEFAULT")
    print(f"manifest_path:      {args.manifest if args.manifest else 'Runtime sampling (ad-hoc)'}")
    print(f"manifest_sha256:    {manifest_sha256 if manifest_sha256 else 'N/A'}")
    print(f"dataloader_workers: {args.nworker}")
    print(f"git_commit:         {git_info.get('git_commit')}")
    print(f"git_status:         {git_info.get('status')}")
    print(f"PyTorch version:    {env_info['torch_version']}")
    print(f"TorchVision version:{env_info['torchvision_version']}")
    print(f"CUDA version:       {env_info['cuda_version']}")
    print(f"GPU:                {env_info['gpu_name']}")
    print(f"Log file saved to:  {log_filepath}")
    print("=" * 80 + "\n")

    # 1. Build Dataloader
    dataloader = build_dataloader(
        benchmark=args.benchmark,
        datapath=args.datapath,
        shot=args.nshot,
        img_size=args.img_size,
        bsz=1,
        nworker=args.nworker,
        split='test',
        manifest_path=args.manifest
    )

    # 2. Initialize Clean CD-FSS Engine
    engine = CDFSSEngine(
        adapter_type=args.adapter,
        fusion_mode=args.fusion,
        fusion_temp=args.fusion_temp,
        adapt_mode=args.adapt_to,
        num_epochs=25,
        lr=1e-2,
        l0=3,
        device=args.device
    )

    # 3. Run Evaluation Loop
    results = engine.evaluate_dataset(dataloader, benchmark_name=args.benchmark, max_episodes=resolved_episodes)

    # Append to master JSON summary
    summary_file = os.path.join(args.logpath, "summary_records.jsonl")
    record = {
        "timestamp": timestamp,
        "experiment": exp_name,
        "benchmark": args.benchmark,
        "adapter": args.adapter,
        "fusion": args.fusion,
        "nshot": args.nshot,
        "episodes": len(results.get('episode_ious', [])),
        "seed": args.seed,
        "Mean_Episode_IoU": round(results.get('Mean_Episode_IoU', results['mIoU']), 2),
        "Cumulative_mIoU": round(results.get('Cumulative_mIoU', results['mIoU']), 2),
        "FB-IoU": round(results['FB-IoU'], 2),
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha256,
        "log_file": log_filepath
    }
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save detailed run report JSON inside experiment directory
    detailed_result_file = os.path.join(args.logpath, "run_result.json")
    full_artifact = {
        "protocol_signature": protocol_sig,
        "code_version": git_info,
        "environment": env_info,
        "config": {
            "experiment": args.experiment,
            "benchmark": args.benchmark,
            "adapter": args.adapter,
            "fusion": args.fusion,
            "fusion_temp": args.fusion_temp,
            "adapt_to": args.adapt_to,
            "num_epochs": 25,
            "lr": 0.01,
            "l0": 3,
            "img_size": args.img_size,
            "seed": args.seed,
            "nshot": args.nshot,
            "manifest_path": args.manifest,
            "manifest_sha256": manifest_sha256
        },
        "seed": args.seed,
        "dataset": args.benchmark,
        "experiment": exp_name,
        "episode_count": len(results.get('episode_ious', [])),
        "metric": {
            "Mean_Episode_IoU": round(results.get('Mean_Episode_IoU', results['mIoU']), 2),
            "Cumulative_mIoU": round(results.get('Cumulative_mIoU', results['mIoU']), 2),
            "FB-IoU": round(results['FB-IoU'], 2),
            "class_ious": results.get('class_ious', {})
        },
        "raw_result": {
            "episode_ious": results.get('episode_ious', []),
            "detailed_episodes": results.get('detailed_episodes', [])
        },
        "summary": record,
        "log": log_filepath
    }
    with open(detailed_result_file, "w", encoding="utf-8") as f:
        json.dump(full_artifact, f, indent=2, ensure_ascii=False)

    sys.stdout = logger.terminal
    print(f"\n[OK] Results recorded to: {summary_file}")
    print(f"[OK] Full artifact saved to: {detailed_result_file}")
    return results

if __name__ == '__main__':
    main()
