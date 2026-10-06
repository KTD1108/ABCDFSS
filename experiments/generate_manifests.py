#!/usr/bin/env python3
"""
Reproducible Episode Manifest Generator for Cross-Domain Few-Shot Segmentation.
Generates deterministic, portable JSON episode manifests for 20, 100, or 1000 episodes
across any of the 5 CD-FSS benchmark datasets.

All image/mask filepaths are stored relative to the dataset root, ensuring
complete portability across Windows, Linux, and cloud environments (Modal GPU).

Usage:
    python experiments/generate_manifests.py --benchmark isic --episodes 100 --seed 42
    python experiments/generate_manifests.py --benchmark all --episodes 1000 --seed 42
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_all_benchmarks import check_or_download_dataset

BENCHMARKS = ['deepglobe', 'isic', 'lung', 'fss', 'suim']
KAGGLE_SLUGS = {
    'isic': 'heyoujue/isic2018-classwise',
    'suim': 'heyoujue/suim-merged',
    'lung': 'heyoujue/lungsegmentation',
    'fss': None,
    'deepglobe': 'heyoujue/deepglobe'
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def make_relative(path: str, base_path: str) -> str:
    """Converts absolute or nested path to portable path relative to base_path."""
    try:
        rel = os.path.relpath(path, base_path)
        return rel.replace('\\', '/')
    except ValueError:
        return path.replace('\\', '/')

def sample_episodes_deepglobe(datapath: str, num_episodes: int, seed: int = 42, shot: int = 1):
    from src.datasets.deepglobe import DeepglobeDataset
    ds = DeepglobeDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = ds.categories
    metadata = ds.img_metadata_classwise
    rng = np.random.RandomState(seed)

    episodes = []
    for ep_id in range(num_episodes):
        cat_idx = ep_id % len(categories)
        cat_name = categories[cat_idx]
        candidates = metadata[cat_name]
        if len(candidates) < 1 + shot:
            chosen = rng.choice(candidates, 1 + shot, replace=True)
        else:
            chosen = rng.choice(candidates, 1 + shot, replace=False)

        q_img = chosen[0]
        s_imgs = chosen[1:]
        q_mask = ds._to_mask_path(q_img)
        s_masks = [ds._to_mask_path(p) for p in s_imgs]

        episodes.append({
            "episode_id": ep_id,
            "class_id": cat_idx,
            "category": cat_name,
            "query_img": make_relative(q_img, base_path),
            "query_mask": make_relative(q_mask, base_path),
            "support_imgs": [make_relative(p, base_path) for p in s_imgs],
            "support_masks": [make_relative(p, base_path) for p in s_masks]
        })
    return episodes

def sample_episodes_isic(datapath: str, num_episodes: int, seed: int = 42, shot: int = 1):
    from src.datasets.isic import ISICDataset
    ds = ISICDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = ds.categories
    metadata = ds.img_metadata_classwise
    input_dir = os.path.join(base_path, 'ISIC2018_Task1-2_Training_Input')
    rng = np.random.RandomState(seed)

    episodes = []
    for ep_id in range(num_episodes):
        cat_idx = ep_id % len(categories)
        cat_name = categories[cat_idx]
        candidates = metadata[cat_name]
        chosen = rng.choice(candidates, 1 + shot, replace=False)
        q_mask_path, s_mask_paths = chosen[0], chosen[1:]

        q_name = os.path.splitext(os.path.basename(q_mask_path))[0].replace('_segmentation', '') + '.jpg'
        q_img_path = os.path.join(input_dir, q_name)

        s_img_paths = []
        for smp in s_mask_paths:
            s_name = os.path.splitext(os.path.basename(smp))[0].replace('_segmentation', '') + '.jpg'
            s_img_paths.append(os.path.join(input_dir, s_name))

        episodes.append({
            "episode_id": ep_id,
            "class_id": cat_idx,
            "category": cat_name,
            "query_img": make_relative(q_img_path, base_path),
            "query_mask": make_relative(q_mask_path, base_path),
            "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
            "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
        })
    return episodes

def sample_episodes_lung(datapath: str, num_episodes: int, seed: int = 42, shot: int = 1):
    from src.datasets.lung import LungDataset
    ds = LungDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    rng = np.random.RandomState(seed)

    episodes = []
    all_indices = list(range(len(ds.img_paths)))
    for ep_id in range(num_episodes):
        chosen = rng.choice(all_indices, 1 + shot, replace=False)
        q_idx, s_indices = chosen[0], chosen[1:]

        q_img_path = ds.img_paths[q_idx]
        q_mask_path = ds.mask_paths[q_idx]
        s_img_paths = [ds.img_paths[i] for i in s_indices]
        s_mask_paths = [ds.mask_paths[i] for i in s_indices]

        episodes.append({
            "episode_id": ep_id,
            "class_id": 0,
            "query_img": make_relative(q_img_path, base_path),
            "query_mask": make_relative(q_mask_path, base_path),
            "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
            "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
        })
    return episodes

def sample_episodes_suim(datapath: str, num_episodes: int, seed: int = 42, shot: int = 1):
    from src.datasets.suim import SUIMDataset
    ds = SUIMDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = ds.categories
    metadata = ds.img_metadata_classwise
    rng = np.random.RandomState(seed)

    episodes = []
    for ep_id in range(num_episodes):
        cat_idx = ep_id % len(categories)
        cat_name = categories[cat_idx]
        candidates = metadata[cat_name]
        chosen = rng.choice(candidates, 1 + shot, replace=False)
        q_mask_path, s_mask_paths = chosen[0], chosen[1:]

        q_img_path = ds._resolve_image_path(q_mask_path)
        s_img_paths = [ds._resolve_image_path(p) for p in s_mask_paths]

        episodes.append({
            "episode_id": ep_id,
            "class_id": cat_idx,
            "category": cat_name,
            "query_img": make_relative(q_img_path, base_path),
            "query_mask": make_relative(q_mask_path, base_path),
            "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
            "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
        })
    return episodes

def sample_episodes_fss(datapath: str, num_episodes: int, seed: int = 42, shot: int = 1):
    from src.datasets.fss import FSS1000Dataset
    ds = FSS1000Dataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    classes = ds.classes
    class_dirs = ds.class_dirs
    rng = np.random.RandomState(seed)

    episodes = []
    for ep_id in range(num_episodes):
        class_name = classes[ep_id % len(classes)]
        class_id = ep_id % len(classes)
        cat_dir = class_dirs[class_name]

        chosen = rng.choice(range(1, 11), 1 + shot, replace=False)
        query_id, s_ids = chosen[0], chosen[1:]
        q_img_path = os.path.join(cat_dir, f"{query_id}.jpg")
        q_mask_path = os.path.join(cat_dir, f"{query_id}.png")
        s_img_paths = [os.path.join(cat_dir, f"{sid}.jpg") for sid in s_ids]
        s_mask_paths = [os.path.join(cat_dir, f"{sid}.png") for sid in s_ids]

        episodes.append({
            "episode_id": ep_id,
            "class_id": class_id,
            "category": class_name,
            "query_img": make_relative(q_img_path, base_path),
            "query_mask": make_relative(q_mask_path, base_path),
            "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
            "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
        })
    return episodes

SAMPLERS = {
    'deepglobe': sample_episodes_deepglobe,
    'isic': sample_episodes_isic,
    'lung': sample_episodes_lung,
    'fss': sample_episodes_fss,
    'fss1000': sample_episodes_fss,
    'suim': sample_episodes_suim,
}

def generate_manifest(benchmark: str, episodes: int = 100, seed: int = 42, shot: int = 1, datapath: str = None, out_file: str = None):
    b = benchmark.lower()
    if b not in SAMPLERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if datapath is None:
        datapath = check_or_download_dataset(b, kaggle_slug=KAGGLE_SLUGS.get(b))

    sampler = SAMPLERS[b]
    ep_list = sampler(datapath, num_episodes=episodes, seed=seed, shot=shot)

    manifest_data = {
        "seed": seed,
        "benchmark": b,
        "nshot": shot,
        "num_episodes": len(ep_list),
        "episodes": ep_list
    }

    if out_file is None:
        out_dir = os.path.join(PROJECT_ROOT, "experiments", "episodes")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{b}_seed{seed}_{episodes}episodes.json")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[OK] Generated manifest for '{b}' ({len(ep_list)} episodes) -> {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Deterministic Episode Manifest Generator for CD-FSS")
    parser.add_argument('--benchmark', type=str, default='all', choices=['all', 'deepglobe', 'isic', 'lung', 'fss', 'suim'])
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes (e.g. 20, 100, 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    parser.add_argument('--nshot', type=int, default=1, help='Number of support shots (default: 1)')
    parser.add_argument('--datapath', type=str, default=None, help='Explicit datapath (optional, otherwise auto-detected)')
    parser.add_argument('--output', type=str, default=None, help='Explicit output path (optional)')
    args = parser.parse_args()

    benchmarks_to_run = BENCHMARKS if args.benchmark == 'all' else [args.benchmark]
    for b in benchmarks_to_run:
        generate_manifest(
            benchmark=b,
            episodes=args.episodes,
            seed=args.seed,
            shot=args.nshot,
            datapath=args.datapath,
            out_file=args.output
        )

if __name__ == '__main__':
    main()
