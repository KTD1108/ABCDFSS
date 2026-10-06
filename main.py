#!/usr/bin/env python3
"""
ABCDFSS Main Entrypoint (Author-Compatible CLI)
Matches the exact usage interface of the original CVPR 2024 paper repository:
    python main.py --benchmark isic --datapath ./datasets/isic/ --nshot 1

Also supports the full E0-E3 architectural ablation suite:
    python main.py --benchmark isic --datapath ./datasets/isic/ --experiment E3 --episodes 100
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate import parse_args, evaluate_pipeline


def main():
    args = parse_args()
    evaluate_pipeline(args)


if __name__ == '__main__':
    main()
