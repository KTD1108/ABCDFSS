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
    parser.add_argument('--adapter', '--adapter-type', dest='adapter', type=str, default='depthwise_separable_3x3',
                        choices=['depthwise_separable_3x3', 'conv1x1'],
                        help='Adapter architecture: depthwise_separable_3x3 or conv1x1')
    parser.add_argument('--fusion', '--fusion-mode', dest='fusion', type=str, default='softmax_margin',
                        choices=['softmax_margin', 'mean', 'learnable'],
                        help='Multi-layer fusion method: softmax_margin or mean')
    parser.add_argument('--fusion-temp', type=float, default=1.0,
                        help='Softmax temperature scaling factor')
    parser.add_argument('--adapt-to', type=str, default='first-episode',
                        choices=['first-episode', 'every-episode'],
                        help='Adaptation mode: first-episode (quick-infer) or every-episode')
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
    from datetime import datetime
    args = parse_args()
    set_seed(args.seed)

    # Setup automatic file logging
    os.makedirs(args.logpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.benchmark}_{args.adapter}_{args.fusion}_{args.nshot}shot_{timestamp}.log"
    log_filepath = os.path.join(args.logpath, log_filename)
    logger = TeeLogger(log_filepath)
    sys.stdout = logger

    print(f"[*] Log file saved to: {log_filepath}")

    # 1. Build Dataloader
    dataloader = build_dataloader(
        benchmark=args.benchmark,
        datapath=args.datapath,
        shot=args.nshot,
        img_size=args.img_size,
        bsz=1,
        nworker=0,
        split='test'
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
    results = engine.evaluate_dataset(dataloader, benchmark_name=args.benchmark)

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
    print(f"\n[✓] Results recorded to: {summary_file}")
    return results

if __name__ == '__main__':
    main()
