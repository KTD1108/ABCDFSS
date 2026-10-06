#!/usr/bin/env python3
"""
Protocol Signature and Audit Validation for ABCDFSS Benchmark Runs.
Enforces rigorous resume criteria, environmental telemetry, and git version tracking.
"""

import os
import sys
import json
import subprocess
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_git_info(repo_dir: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Retrieves current git commit and clean/dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return {
            "git_commit": commit,
            "git_dirty": len(status_out) > 0,
            "status": "dirty" if len(status_out) > 0 else "clean"
        }
    except Exception as e:
        env_commit = os.environ.get("ABCDFSS_GIT_COMMIT")
        if env_commit:
            return {
                "git_commit": env_commit,
                "git_dirty": os.environ.get("ABCDFSS_GIT_DIRTY", "false").lower() == "true",
                "status": "injected_via_env"
            }
        return {
            "git_commit": "unknown",
            "git_dirty": None,
            "reason": f"Git metadata not accessible in runtime environment ({type(e).__name__})"
        }

def get_environment_info() -> Dict[str, Any]:
    """Captures runtime hardware and deep learning package versions."""
    import torch
    import torchvision
    import numpy
    import PIL
    import scipy

    cuda_avail = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": numpy.__version__,
        "pillow_version": PIL.__version__,
        "scipy_version": scipy.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU Only",
        "platform": sys.platform
    }

def create_protocol_signature(
    benchmark: str,
    experiment: str,
    episodes: Any,
    seed: int,
    nshot: int = 1,
    adapt_to: str = "every-episode",
    adapter: str = "conv1x1",
    fusion: str = "mean",
    fusion_temp: float = 1.0,
    image_size: int = 400,
    num_epochs: int = 25,
    learning_rate: float = 0.01,
    manifest: Optional[str] = None,
    manifest_sha256: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a deterministic dictionary defining the exact scientific protocol.
    Used to validate resume decisions and guarantee historical non-falsification.
    """
    return {
        "benchmark": benchmark.lower(),
        "experiment": experiment.upper(),
        "episodes": "all" if str(episodes).lower() == "all" else int(episodes),
        "seed": int(seed),
        "nshot": int(nshot),
        "adapt_to": adapt_to,
        "adapter": adapter,
        "fusion": fusion,
        "fusion_temp": float(fusion_temp),
        "image_size": int(image_size),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256
    }

def validate_protocol_signature(
    saved_sig: Optional[Dict[str, Any]],
    expected_sig: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Strict resume validation.
    Returns (True, "OK") ONLY if the previous result's full protocol signature matches
    the current requested run. If any important field differs, returns (False, reason).
    """
    if not saved_sig or not isinstance(saved_sig, dict):
        return False, "Missing or invalid protocol signature in saved artifact"

    critical_fields = [
        "benchmark", "experiment", "episodes", "seed", "nshot",
        "adapt_to", "adapter", "fusion", "image_size", "num_epochs",
        "learning_rate", "manifest_sha256"
    ]

    for field in critical_fields:
        if field not in saved_sig:
            return False, f"Saved signature missing required field '{field}'"

        saved_val = saved_sig[field]
        exp_val = expected_sig.get(field)

        # Allow string/int comparison for episodes if normalized
        if field == "episodes":
            if str(saved_val).lower() != str(exp_val).lower():
                return False, f"Episode count mismatch: saved '{saved_val}' vs requested '{exp_val}'"
            continue

        if saved_val != exp_val:
            return False, f"Mismatch in protocol field '{field}': saved '{saved_val}' vs requested '{exp_val}'"

    return True, "OK"
