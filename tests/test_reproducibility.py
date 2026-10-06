#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for ABCDFSS Manifests, Protocols, and Reproducibility Infrastructure.
Covers all 14 requirements from the Scientific Audit Specification.
"""

import os
import sys
import json
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.manifest import (
    resolve_manifest_path,
    validate_manifest,
    compute_manifest_sha256,
    normalize_benchmark_name
)
from src.utils.protocol import (
    create_protocol_signature,
    validate_protocol_signature,
    get_git_info,
    get_environment_info
)

class TestReproducibilityPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.episodes_dir = os.path.join(self.temp_dir.name, "experiments", "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_manifest(self, benchmark="isic", seed=42, episodes_count=5, filename=None):
        episodes = []
        for i in range(episodes_count):
            episodes.append({
                "episode_id": i,
                "class_id": i % 3,
                "category": str(i % 3 + 1),
                "query_img": f"images/q_{i}.jpg",
                "query_mask": f"masks/q_{i}.png",
                "support_imgs": [f"images/s_{i}.jpg"],
                "support_masks": [f"masks/s_{i}.png"]
            })
        data = {
            "seed": seed,
            "benchmark": benchmark,
            "nshot": 1,
            "num_episodes": episodes_count,
            "episodes": episodes
        }
        if filename is None:
            filename = f"{benchmark}_seed{seed}_{episodes_count}episodes.json"
        path = os.path.join(self.episodes_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    # 1. Manifest path resolution
    def test_01_manifest_path_resolution(self):
        path = self._create_dummy_manifest(benchmark="deepglobe", seed=42, episodes_count=20)
        resolved = resolve_manifest_path("deepglobe", seed=42, episodes=20, base_dir=self.episodes_dir)
        self.assertEqual(resolved, os.path.abspath(path))

    # 2. Manifest validation
    def test_02_manifest_validation(self):
        path = self._create_dummy_manifest(benchmark="isic", seed=42, episodes_count=10)
        info = validate_manifest(path, benchmark="isic", seed=42, episodes=10)
        self.assertTrue(info["valid"])
        self.assertEqual(info["resolved_episodes"], 10)
        self.assertEqual(info["seed"], 42)
        self.assertEqual(len(info["manifest_sha256"]), 64)

    # 3. --episodes 20 resolution
    def test_03_episodes_20_resolution(self):
        path = self._create_dummy_manifest(benchmark="lung", seed=42, episodes_count=20)
        resolved = resolve_manifest_path("lung", seed=42, episodes="20", base_dir=self.episodes_dir)
        self.assertEqual(resolved, os.path.abspath(path))
        val = validate_manifest(resolved, benchmark="lung", episodes=20)
        self.assertEqual(val["resolved_episodes"], 20)

    # 4. --episodes 100 resolution
    def test_04_episodes_100_resolution(self):
        path = self._create_dummy_manifest(benchmark="suim", seed=42, episodes_count=100)
        resolved = resolve_manifest_path("suim", seed=42, episodes="100", base_dir=self.episodes_dir)
        self.assertEqual(resolved, os.path.abspath(path))
        val = validate_manifest(resolved, benchmark="suim", episodes=100)
        self.assertEqual(val["resolved_episodes"], 100)

    # 5. --episodes 1000 resolution
    def test_05_episodes_1000_resolution(self):
        path = self._create_dummy_manifest(benchmark="fss", seed=42, episodes_count=1000)
        resolved = resolve_manifest_path("fss", seed=42, episodes="1000", base_dir=self.episodes_dir)
        self.assertEqual(resolved, os.path.abspath(path))
        val = validate_manifest(resolved, benchmark="fss", episodes=1000)
        self.assertEqual(val["resolved_episodes"], 1000)

    # 6. --episodes all resolution
    def test_06_episodes_all_resolution(self):
        path = self._create_dummy_manifest(benchmark="deepglobe", seed=42, episodes_count=15, filename="deepglobe_seed42_all_episodes.json")
        resolved = resolve_manifest_path("deepglobe", seed=42, episodes="all", base_dir=self.episodes_dir)
        self.assertEqual(resolved, os.path.abspath(path))
        val = validate_manifest(resolved, benchmark="deepglobe", episodes="all")
        self.assertEqual(val["resolved_episodes"], 15)

    # 7. Same manifest used across E0-E3
    def test_07_same_manifest_used_by_e0_e3(self):
        manifest_path = self._create_dummy_manifest(benchmark="isic", seed=42, episodes_count=50)
        val = validate_manifest(manifest_path, benchmark="isic", seed=42, episodes=50)

        manifests_for_exps = []
        sha_for_exps = []
        for exp in ['E0', 'E1', 'E2', 'E3']:
            m = resolve_manifest_path("isic", seed=42, episodes=50, base_dir=self.episodes_dir)
            manifests_for_exps.append(m)
            sha_for_exps.append(compute_manifest_sha256(m))

        self.assertEqual(len(set(manifests_for_exps)), 1, "E0-E3 did not use identical manifest path")
        self.assertEqual(len(set(sha_for_exps)), 1, "E0-E3 did not use identical manifest SHA256")
        self.assertEqual(sha_for_exps[0], val["manifest_sha256"])

    # 8. Missing manifest produces clear failure
    def test_08_missing_manifest_produces_clear_failure(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_manifest_path("lung", seed=42, episodes=999, base_dir=self.episodes_dir, allow_missing=False)
        err_msg = str(ctx.exception)
        self.assertIn("No deterministic episode manifest found", err_msg)
        self.assertIn("python experiments/generate_manifests.py", err_msg)

    # 9. Invalid manifest produces clear failure
    def test_09_invalid_manifest_produces_clear_failure(self):
        # A. Non-unique episode IDs
        bad_path = os.path.join(self.episodes_dir, "bad_manifest.json")
        bad_data = {
            "seed": 42, "benchmark": "isic",
            "episodes": [
                {"episode_id": 0, "query_img": "q.jpg", "query_mask": "q.png", "support_imgs": ["s.jpg"], "support_masks": ["s.png"]},
                {"episode_id": 0, "query_img": "q2.jpg", "query_mask": "q2.png", "support_imgs": ["s2.jpg"], "support_masks": ["s2.png"]}
            ]
        }
        with open(bad_path, "w") as f:
            json.dump(bad_data, f)

        with self.assertRaises(ValueError) as ctx:
            validate_manifest(bad_path, benchmark="isic")
        self.assertIn("Duplicate episode_id", str(ctx.exception))

        # B. Missing required fields
        bad_data2 = {
            "seed": 42, "benchmark": "isic",
            "episodes": [{"episode_id": 0, "query_img": "q.jpg"}]
        }
        with open(bad_path, "w") as f:
            json.dump(bad_data2, f)
        with self.assertRaises(ValueError) as ctx2:
            validate_manifest(bad_path, benchmark="isic")
        self.assertIn("missing required field", str(ctx2.exception))

    # 10. Resume rejects mismatched seed
    def test_10_resume_rejects_mismatched_seed(self):
        sig1 = create_protocol_signature("lung", "E0", 20, seed=42, manifest_sha256="abc123sha")
        sig2 = create_protocol_signature("lung", "E0", 20, seed=99, manifest_sha256="abc123sha")
        match, reason = validate_protocol_signature(sig1, sig2)
        self.assertFalse(match)
        self.assertIn("seed", reason)

    # 11. Resume rejects mismatched manifest SHA256
    def test_11_resume_rejects_mismatched_manifest_sha256(self):
        sig1 = create_protocol_signature("lung", "E0", 20, seed=42, manifest_sha256="hash_version_1")
        sig2 = create_protocol_signature("lung", "E0", 20, seed=42, manifest_sha256="hash_version_2")
        match, reason = validate_protocol_signature(sig1, sig2)
        self.assertFalse(match)
        self.assertIn("manifest_sha256", reason)

    # 12. Resume rejects mismatched adapter/fusion
    def test_12_resume_rejects_mismatched_adapter_or_fusion(self):
        sig_e0 = create_protocol_signature("lung", "E0", 20, seed=42, adapter="conv1x1", fusion="mean")
        sig_e1 = create_protocol_signature("lung", "E1", 20, seed=42, adapter="depthwise_separable_3x3", fusion="mean")
        match, reason = validate_protocol_signature(sig_e0, sig_e1)
        self.assertFalse(match)

        sig_e2 = create_protocol_signature("lung", "E2", 20, seed=42, adapter="conv1x1", fusion="softmax_margin")
        match2, reason2 = validate_protocol_signature(sig_e0, sig_e2)
        self.assertFalse(match2)

    # 13. Modal manifest resolution
    def test_13_modal_manifest_resolution(self):
        modal_dir = os.path.join(self.temp_dir.name, "root", "ABCDFSS", "experiments", "episodes")
        os.makedirs(modal_dir, exist_ok=True)
        m_path = os.path.join(modal_dir, "deepglobe_seed42_100episodes.json")
        with open(m_path, "w") as f:
            json.dump({"benchmark": "deepglobe", "seed": 42, "episodes": [{"episode_id": 0, "query_img": "a", "query_mask": "b", "support_imgs": ["c"], "support_masks": ["d"]}]}, f)

        res = resolve_manifest_path("deepglobe", seed=42, episodes=100, base_dir=modal_dir)
        self.assertEqual(res, os.path.abspath(m_path))

    # 14. Protocol signature generation
    def test_14_protocol_signature_generation(self):
        sig = create_protocol_signature(
            benchmark="isic",
            experiment="E3",
            episodes=1000,
            seed=42,
            nshot=1,
            adapt_to="every-episode",
            adapter="depthwise_separable_3x3",
            fusion="softmax_margin",
            fusion_temp=1.0,
            image_size=400,
            num_epochs=25,
            learning_rate=0.01,
            manifest="/path/to/manifest.json",
            manifest_sha256="deadbeef12345678"
        )
        self.assertEqual(sig["benchmark"], "isic")
        self.assertEqual(sig["experiment"], "E3")
        self.assertEqual(sig["adapter"], "depthwise_separable_3x3")
        self.assertEqual(sig["fusion"], "softmax_margin")
        self.assertEqual(sig["num_epochs"], 25)
        self.assertEqual(sig["learning_rate"], 0.01)
        self.assertEqual(sig["manifest_sha256"], "deadbeef12345678")

        # Self-match must pass
        match, reason = validate_protocol_signature(sig, sig)
        self.assertTrue(match)

if __name__ == '__main__':
    unittest.main()
