#!/usr/bin/env python3
"""
Manifest Resolution and Validation Utilities for Cross-Domain Few-Shot Segmentation.
Ensures deterministic, auditable, and portable episode selection across all environments.
"""

import os
import sys
import json
import hashlib
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_manifest_sha256(manifest_path: str) -> str:
    """Computes SHA-256 hash of a manifest file for cryptographic protocol validation."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found for hashing: {manifest_path}")
    hasher = hashlib.sha256()
    with open(manifest_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_benchmark_name(benchmark: str) -> str:
    """Normalizes benchmark name aliases (e.g. fss1000 -> fss)."""
    b = benchmark.lower().strip()
    if b in ['fss', 'fss1000', 'fss-1000', 'fewshot1000']:
        return 'fss'
    if b in ['lung', 'chest', 'xray', 'lungsegmentation']:
        return 'lung'
    if b in ['isic', 'isic2018']:
        return 'isic'
    if b in ['deepglobe']:
        return 'deepglobe'
    if b in ['suim', 'suim-merged']:
        return 'suim'
    return b

def resolve_manifest_path(
    benchmark: str,
    seed: int = 42,
    episodes: Any = 20,
    manifest_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    allow_missing: bool = False
) -> Optional[str]:
    """
    Deterministically resolves the episode manifest path.
    If --manifest is supplied explicitly, validates its existence.
    Otherwise, automatically looks up experiments/episodes/{benchmark}_seed{seed}_{episodes}episodes.json.
    Raises FileNotFoundError with clear instructions if missing in benchmark mode.
    """
    b_norm = normalize_benchmark_name(benchmark)
    ep_str = str(episodes).lower().strip()

    # 1. Explicit manifest supplied
    if manifest_path:
        cand = manifest_path if os.path.isabs(manifest_path) else os.path.join(PROJECT_ROOT, manifest_path)
        if os.path.exists(cand):
            return os.path.abspath(cand)
        raise FileNotFoundError(f"[ERROR] Explicitly specified manifest does not exist: {manifest_path} (resolved: {cand})")

    # 2. Automated resolution in experiments/episodes
    episodes_dir = base_dir or os.path.join(PROJECT_ROOT, "experiments", "episodes")

    if ep_str == 'all':
        candidate = os.path.join(episodes_dir, f"{b_norm}_seed{seed}_all_episodes.json")
    else:
        candidate = os.path.join(episodes_dir, f"{b_norm}_seed{seed}_{ep_str}episodes.json")

    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # Secondary lookup: check if single manifest exists matching benchmark and seed
    if os.path.exists(episodes_dir):
        matching = [
            os.path.join(episodes_dir, f) for f in os.listdir(episodes_dir)
            if f.startswith(f"{b_norm}_seed{seed}_") and f.endswith(".json")
        ]
        # If ep_str == 'all' and there is a manifest with all episodes or exact match
        if ep_str == 'all' and len(matching) == 1:
            return os.path.abspath(matching[0])

    if allow_missing:
        return None

    # Clear, descriptive failure for reproducible benchmark mode
    raise FileNotFoundError(
        f"\n================================================================================\n"
        f"[ERROR] No deterministic episode manifest found for benchmark run:\n"
        f"  Benchmark: {benchmark} (normalized: {b_norm})\n"
        f"  Seed:      {seed}\n"
        f"  Episodes:  {episodes}\n"
        f"  Expected:  {candidate}\n\n"
        f"ABCDFSS requires explicit, deterministic manifests for reproducible evaluation.\n"
        f"Generate this manifest deterministically before running the benchmark:\n"
        f"  python experiments/generate_manifests.py --benchmark {b_norm} --episodes {ep_str} --seed {seed}\n"
        f"================================================================================\n"
    )

def validate_manifest(
    manifest_path: str,
    benchmark: Optional[str] = None,
    seed: Optional[int] = None,
    episodes: Optional[Any] = None,
    nshot: Optional[int] = None,
    dataset_base_path: Optional[str] = None,
    check_files_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive structural and integrity validation on an episode manifest:
    1. File existence
    2. Valid JSON
    3. Non-empty episodes list
    4. Unique & non-null episode IDs (strictly checks for missing and duplicate IDs)
    5. Required fields per episode
    6. Exhaustive image & mask file existence on disk (all episodes verified if dataset_base_path provided)
    7. Consistent query/support structure and support size matching nshot
    8. Benchmark alignment
    9. Seed alignment
    10. Episode count validation (or 'all' resolution)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"[Manifest Error] Manifest file does not exist: {manifest_path}")

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"[Manifest Error] Invalid JSON in {manifest_path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"[Manifest Error] Manifest root must be a JSON object, got {type(data).__name__}")

    if "episodes" not in data or not isinstance(data["episodes"], list):
        raise ValueError(f"[Manifest Error] Manifest missing 'episodes' list in {manifest_path}")

    ep_list = data["episodes"]
    if len(ep_list) == 0:
        raise ValueError(f"[Manifest Error] Manifest contains 0 episodes in {manifest_path}")

    # Check num_episodes header consistency if present
    if "num_episodes" in data and data["num_episodes"] != len(ep_list):
        raise ValueError(
            f"[Manifest Error] Manifest header num_episodes ({data['num_episodes']}) "
            f"does not match actual episodes count ({len(ep_list)}) in {manifest_path}"
        )

    # Validate uniqueness and presence of episode IDs
    seen_ids = set()
    for idx, ep in enumerate(ep_list):
        if not isinstance(ep, dict):
            raise ValueError(f"[Manifest Error] Episode at index {idx} is not an object in {manifest_path}")
        if "episode_id" not in ep or ep["episode_id"] is None:
            raise ValueError(f"[Manifest Error] Missing episode_id at index {idx} in {manifest_path}")
        ep_id = ep["episode_id"]
        if ep_id in seen_ids:
            raise ValueError(f"[Manifest Error] Duplicate episode_id {ep_id} detected in {manifest_path}")
        seen_ids.add(ep_id)

        # Check required fields
        required_fields = ["query_img", "query_mask", "support_imgs", "support_masks"]
        for rf in required_fields:
            if rf not in ep:
                raise ValueError(f"[Manifest Error] Episode {ep_id} missing required field '{rf}' in {manifest_path}")

        # Check query and support data types
        if not isinstance(ep["support_imgs"], list) or not isinstance(ep["support_masks"], list):
            raise ValueError(f"[Manifest Error] Episode {ep_id} support_imgs/masks must be lists")
        if len(ep["support_imgs"]) != len(ep["support_masks"]):
            raise ValueError(f"[Manifest Error] Episode {ep_id} support_imgs count != support_masks count")
        if len(ep["support_imgs"]) == 0:
            raise ValueError(f"[Manifest Error] Episode {ep_id} has empty support set")
        if nshot is not None and len(ep["support_imgs"]) != nshot:
            raise ValueError(f"[Manifest Error] Episode {ep_id} support set size ({len(ep['support_imgs'])}) != expected nshot ({nshot})")

    # Validate file existence on disk if dataset base path is provided
    # For full dataset / benchmark runs (check_files_limit is None), validates 100% of files
    if dataset_base_path and os.path.exists(dataset_base_path):
        effective_base = dataset_base_path
        if not os.path.exists(os.path.join(effective_base, 'images')) and os.path.exists(os.path.join(effective_base, 'suim_merged')):
            effective_base = os.path.join(effective_base, 'suim_merged')
        elif not os.path.exists(os.path.join(effective_base, 'fewshot_data')) and os.path.exists(os.path.join(effective_base, 'fewshot1000')):
            effective_base = os.path.join(effective_base, 'fewshot1000')

        is_all_mode = episodes is not None and str(episodes).lower().strip() == 'all'
        sample_to_check = ep_list if (check_files_limit is None or is_all_mode) else ep_list[:check_files_limit]
        for ep in sample_to_check:
            # Query img & mask (normalize backslashes for cross-platform portability)
            q_img_rel = ep["query_img"].replace('\\', '/')
            q_mask_rel = ep["query_mask"].replace('\\', '/')
            q_img = q_img_rel if os.path.isabs(q_img_rel) else os.path.join(effective_base, q_img_rel)
            q_mask = q_mask_rel if os.path.isabs(q_mask_rel) else os.path.join(effective_base, q_mask_rel)
            if not os.path.exists(q_img):
                raise FileNotFoundError(f"[Manifest Error] Query image does not exist: {q_img} (episode {ep.get('episode_id')})")
            if not os.path.exists(q_mask):
                raise FileNotFoundError(f"[Manifest Error] Query mask does not exist: {q_mask} (episode {ep.get('episode_id')})")

            # Support imgs & masks
            for s_img_rel_raw in ep["support_imgs"]:
                s_img_rel = s_img_rel_raw.replace('\\', '/')
                s_img = s_img_rel if os.path.isabs(s_img_rel) else os.path.join(effective_base, s_img_rel)
                if not os.path.exists(s_img):
                    raise FileNotFoundError(f"[Manifest Error] Support image does not exist: {s_img} (episode {ep.get('episode_id')})")
            for s_mask_rel_raw in ep["support_masks"]:
                s_mask_rel = s_mask_rel_raw.replace('\\', '/')
                s_mask = s_mask_rel if os.path.isabs(s_mask_rel) else os.path.join(effective_base, s_mask_rel)
                if not os.path.exists(s_mask):
                    raise FileNotFoundError(f"[Manifest Error] Support mask does not exist: {s_mask} (episode {ep.get('episode_id')})")

    # Benchmark name validation
    if benchmark:
        b_norm = normalize_benchmark_name(benchmark)
        m_b = normalize_benchmark_name(data.get("benchmark", b_norm))
        if b_norm != m_b:
            raise ValueError(f"[Manifest Error] Manifest benchmark mismatch: requested '{b_norm}', manifest has '{m_b}'")

    # Seed validation
    if seed is not None and "seed" in data:
        if data["seed"] != seed:
            raise ValueError(f"[Manifest Error] Manifest seed mismatch: requested seed={seed}, manifest has seed={data['seed']}")

    # N-shot validation in header
    if nshot is not None and "nshot" in data:
        if data["nshot"] != nshot:
            raise ValueError(f"[Manifest Error] Manifest nshot mismatch: requested nshot={nshot}, manifest header has nshot={data['nshot']}")

    # Episode count validation
    total_in_manifest = len(ep_list)
    resolved_episodes = total_in_manifest

    if episodes is not None:
        ep_str = str(episodes).lower().strip()
        if ep_str == 'all':
            resolved_episodes = total_in_manifest
        else:
            req_int = int(episodes)
            if total_in_manifest != req_int:
                raise ValueError(
                    f"[Manifest Error] Manifest episode count mismatch in {manifest_path}: "
                    f"requested {req_int} episodes, but manifest contains {total_in_manifest} episodes."
                )
            resolved_episodes = req_int

    sha256 = compute_manifest_sha256(manifest_path)
    return {
        "valid": True,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": sha256,
        "total_episodes": total_in_manifest,
        "resolved_episodes": resolved_episodes,
        "benchmark": data.get("benchmark", benchmark),
        "seed": data.get("seed", seed),
        "nshot": data.get("nshot", nshot or 1),
        "mode": data.get("mode", "exhaustive_full_dataset" if episodes == "all" else "sampled"),
        "episodes": ep_list
    }
