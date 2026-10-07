#!/usr/bin/env python3
"""
Reproducible Episode Manifest Generator for Cross-Domain Few-Shot Segmentation.
Supports both:
1. Deterministic sampled evaluation (e.g. --episodes 20, 100, 1000)
2. Exhaustive full-dataset evaluation (--episodes all), where every valid query image
   appears exactly once, and support images are selected deterministically with seed.

All image/mask filepaths are stored relative to the dataset root, ensuring
complete portability across Windows, Linux, and cloud environments (Modal GPU).

Usage:
    python experiments/generate_manifests.py --benchmark isic --episodes 100 --seed 42
    python experiments/generate_manifests.py --benchmark all --episodes 1000 --seed 42
    python experiments/generate_manifests.py --benchmark all --episodes all --seed 42
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from typing import Any, List, Dict

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

def sample_episodes_deepglobe(datapath: str, num_episodes: Any, seed: int = 42, shot: int = 1) -> List[Dict[str, Any]]:
    from src.datasets.deepglobe import DeepglobeDataset
    ds = DeepglobeDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = sorted(ds.categories)
    metadata = ds.img_metadata_classwise
    rng = np.random.RandomState(seed)

    ep_str = str(num_episodes).lower().strip()
    episodes = []

    if ep_str == 'all':
        # Exhaustive mode: each query image appears exactly once
        ep_id = 0
        for cat_idx, cat_name in enumerate(categories):
            candidates = sorted(metadata[cat_name])
            for q_img in candidates:
                available = [p for p in candidates if p != q_img]
                if len(available) < shot:
                    s_imgs = rng.choice(available, shot, replace=True)
                else:
                    s_imgs = rng.choice(available, shot, replace=False)
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
                ep_id += 1
    else:
        # Sampled mode
        req_count = int(num_episodes)
        for ep_id in range(req_count):
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

def sample_episodes_isic(datapath: str, num_episodes: Any, seed: int = 42, shot: int = 1) -> List[Dict[str, Any]]:
    from src.datasets.isic import ISICDataset
    ds = ISICDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = sorted(ds.categories)
    metadata = ds.img_metadata_classwise
    input_dir = os.path.join(base_path, 'ISIC2018_Task1-2_Training_Input')
    rng = np.random.RandomState(seed)

    ep_str = str(num_episodes).lower().strip()
    episodes = []

    if ep_str == 'all':
        # Exhaustive mode: each query image appears exactly once
        ep_id = 0
        for cat_idx, cat_name in enumerate(categories):
            candidates = sorted(metadata[cat_name])
            for q_mask_path in candidates:
                q_name = os.path.splitext(os.path.basename(q_mask_path))[0].replace('_segmentation', '') + '.jpg'
                q_img_path = os.path.join(input_dir, q_name)

                available = [p for p in candidates if p != q_mask_path]
                s_mask_paths = rng.choice(available, shot, replace=False)

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
                ep_id += 1
    else:
        req_count = int(num_episodes)
        for ep_id in range(req_count):
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

def sample_episodes_lung(datapath: str, num_episodes: Any, seed: int = 42, shot: int = 1) -> List[Dict[str, Any]]:
    from src.datasets.lung import LungDataset
    ds = LungDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    rng = np.random.RandomState(seed)

    ep_str = str(num_episodes).lower().strip()
    episodes = []

    if ep_str == 'all':
        # Exhaustive mode: every image in Lung appears exactly once
        indices = sorted(range(len(ds.img_paths)))
        for ep_id, q_idx in enumerate(indices):
            q_img_path = ds.img_paths[q_idx]
            q_mask_path = ds.mask_paths[q_idx]

            available = [i for i in indices if i != q_idx]
            s_indices = rng.choice(available, shot, replace=False)
            s_img_paths = [ds.img_paths[i] for i in s_indices]
            s_mask_paths = [ds.mask_paths[i] for i in s_indices]

            episodes.append({
                "episode_id": ep_id,
                "class_id": 0,
                "category": "lung",
                "query_img": make_relative(q_img_path, base_path),
                "query_mask": make_relative(q_mask_path, base_path),
                "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
                "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
            })
    else:
        req_count = int(num_episodes)
        all_indices = list(range(len(ds.img_paths)))
        for ep_id in range(req_count):
            chosen = rng.choice(all_indices, 1 + shot, replace=False)
            q_idx, s_indices = chosen[0], chosen[1:]

            q_img_path = ds.img_paths[q_idx]
            q_mask_path = ds.mask_paths[q_idx]
            s_img_paths = [ds.img_paths[i] for i in s_indices]
            s_mask_paths = [ds.mask_paths[i] for i in s_indices]

            episodes.append({
                "episode_id": ep_id,
                "class_id": 0,
                "category": "lung",
                "query_img": make_relative(q_img_path, base_path),
                "query_mask": make_relative(q_mask_path, base_path),
                "support_imgs": [make_relative(p, base_path) for p in s_img_paths],
                "support_masks": [make_relative(p, base_path) for p in s_mask_paths]
            })
    return episodes

def sample_episodes_suim(datapath: str, num_episodes: Any, seed: int = 42, shot: int = 1) -> List[Dict[str, Any]]:
    from src.datasets.suim import SUIMDataset
    ds = SUIMDataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    categories = sorted(ds.categories)
    metadata = ds.img_metadata_classwise
    rng = np.random.RandomState(seed)

    ep_str = str(num_episodes).lower().strip()
    episodes = []

    if ep_str == 'all':
        # Exhaustive mode: every mask in SUIM appears exactly once
        ep_id = 0
        for cat_idx, cat_name in enumerate(categories):
            candidates = sorted(metadata[cat_name])
            for q_mask_path in candidates:
                q_img_path = ds._resolve_image_path(q_mask_path)
                available = [p for p in candidates if p != q_mask_path]
                s_mask_paths = rng.choice(available, shot, replace=False)
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
                ep_id += 1
    else:
        req_count = int(num_episodes)
        for ep_id in range(req_count):
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

def sample_episodes_fss(datapath: str, num_episodes: Any, seed: int = 42, shot: int = 1) -> List[Dict[str, Any]]:
    from src.datasets.fss import FSS1000Dataset
    ds = FSS1000Dataset(datapath, transform=lambda x: x, shot=shot)
    base_path = ds.base_path
    classes = sorted(ds.classes)
    class_dirs = ds.class_dirs
    rng = np.random.RandomState(seed)

    ep_str = str(num_episodes).lower().strip()
    episodes = []

    if ep_str == 'all':
        # Exhaustive mode: all 10 images of all classes appear exactly once as query
        ep_id = 0
        for class_id, class_name in enumerate(classes):
            cat_dir = class_dirs[class_name]
            for query_id in range(1, 11):
                q_img_path = os.path.join(cat_dir, f"{query_id}.jpg")
                q_mask_path = os.path.join(cat_dir, f"{query_id}.png")

                available = [sid for sid in range(1, 11) if sid != query_id]
                s_ids = rng.choice(available, shot, replace=False)
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
                ep_id += 1
    else:
        req_count = int(num_episodes)
        for ep_id in range(req_count):
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

def generate_manifest(benchmark: str, episodes: Any = 100, seed: int = 42, shot: int = 1, datapath: str = None, out_file: str = None) -> str:
    b = benchmark.lower()
    if b not in SAMPLERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if datapath is None:
        datapath = check_or_download_dataset(b, kaggle_slug=KAGGLE_SLUGS.get(b))

    sampler = SAMPLERS[b]
    ep_str = str(episodes).lower().strip()
    ep_list = sampler(datapath, num_episodes=ep_str if ep_str == 'all' else int(episodes), seed=seed, shot=shot)

    manifest_data = {
        "seed": seed,
        "benchmark": b,
        "nshot": shot,
        "mode": "exhaustive_full_dataset" if ep_str == "all" else "sampled",
        "num_episodes": len(ep_list),
        "episodes": ep_list
    }

    if out_file is None:
        out_dir = os.path.join(PROJECT_ROOT, "experiments", "episodes")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{b}_seed{seed}_all_episodes.json" if ep_str == 'all' else f"{b}_seed{seed}_{len(ep_list)}episodes.json"
        out_file = os.path.join(out_dir, filename)

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[OK] Generated manifest for '{b}' ({len(ep_list)} episodes, mode={manifest_data['mode']}) -> {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Deterministic & Exhaustive Episode Manifest Generator for CD-FSS")
    parser.add_argument('--benchmark', type=str, default='all', choices=['all', 'deepglobe', 'isic', 'lung', 'fss', 'suim'])
    parser.add_argument('--episodes', type=str, default='100', help='Number of episodes (e.g. 20, 100, 1000, or "all" for exhaustive full-dataset evaluation)')
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
