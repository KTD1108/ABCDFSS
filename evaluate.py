#!/usr/bin/env python3
"""
Cross-Domain Few-Shot Segmentation (CD-FSS) Evaluation
Independent, clean self-built framework.

Usage examples:
    python evaluate.py --benchmark isic --datapath /path/to/isic --adapter depthwise_separable_3x3 --fusion softmax_margin
    python evaluate.py --benchmark suim --datapath /path/to/suim --adapter conv1x1 --fusion softmax_margin
"""

import sys
import os

# Ensure project root is always in sys.path for direct or subprocess invocation
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
import random
import numpy as np

from src.datasets import build_dataloader
from src.engine import CDFSSEngine

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
    parser.add_argument('--episodes', type=int, default=None,
                        help='Maximum number of episodes to evaluate (e.g. 1000 for standard CVPR benchmark)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to pre-generated episode manifest JSON')
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
    return parser.parse_args()

class TeeLogger:
    def __init__(self, filepath: str):
        import sys
        self.terminal = sys.stdout
        import os
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
    import sys, os, json
    import torchvision
    from datetime import datetime
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

    # Auto-detect default manifest if exists and episodes == 20
    if args.manifest is None:
        candidate_manifest = os.path.join(PROJECT_ROOT, "experiments", "episodes", f"{args.benchmark}_seed42_20episodes.json")
        if os.path.exists(candidate_manifest) and (args.episodes == 20 or args.episodes is None):
            args.manifest = candidate_manifest

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

    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"

    print("=" * 80)
    print("                     CD-FSS EXPERIMENTAL AUDIT RUNNER")
    print("=" * 80)
    print(f"experiment_name:    {exp_name}")
    print(f"dataset:            {args.benchmark}")
    print(f"seed:               {args.seed}")
    print(f"nshot:              {args.nshot}")
    print(f"num_episodes:       {args.episodes if args.episodes else 'All'}")
    print(f"adapter:            {args.adapter}")
    print(f"fusion:             {args.fusion}")
    print(f"adaptation_mode:    {args.adapt_to}")
    print(f"image_size:         {args.img_size}x{args.img_size}")
    print(f"threshold_method:   pred_mean (max(Otsu, mean), drop_least=0.05)")
    print(f"backbone:           ResNet-50 (Pre-ReLU unclipped features)")
    print(f"checkpoint:         ResNet50_Weights.DEFAULT")
    print(f"manifest_path:      {args.manifest if args.manifest else 'Runtime sampling'}")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"TorchVision version:{torchvision.__version__}")
    print(f"CUDA version:       {cuda_ver}")
    print(f"GPU:                {gpu_name}")
    print(f"Log file saved to:  {log_filepath}")
    print("=" * 80 + "\n")

    # 1. Build Dataloader
    dataloader = build_dataloader(
        benchmark=args.benchmark,
        datapath=args.datapath,
        shot=args.nshot,
        img_size=args.img_size,
        bsz=1,
        nworker=0,
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
    results = engine.evaluate_dataset(dataloader, benchmark_name=args.benchmark, max_episodes=args.episodes)

    # Append to master JSON summary
    summary_file = os.path.join(args.logpath, "summary_records.jsonl")
    record = {
        "timestamp": timestamp,
        "benchmark": args.benchmark,
        "adapter": args.adapter,
        "fusion": args.fusion,
        "nshot": args.nshot,
        "mIoU": round(results['mIoU'], 2),
        "FB-IoU": round(results['FB-IoU'], 2),
        "log_file": log_filepath
    }
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sys.stdout = logger.terminal
    print(f"\n[OK] Results recorded to: {summary_file}")
    return results

if __name__ == '__main__':
    main()
