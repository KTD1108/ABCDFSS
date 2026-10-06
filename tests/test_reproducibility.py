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

    # 15. Protocol signature rejects mismatched fusion_temp (P1.4)
    def test_15_protocol_signature_rejects_mismatched_fusion_temp(self):
        sig1 = create_protocol_signature("isic", "E2", 20, seed=42, adapter="conv1x1", fusion="softmax_margin", fusion_temp=1.0, manifest_sha256="dummy_hash")
        sig2 = create_protocol_signature("isic", "E2", 20, seed=42, adapter="conv1x1", fusion="softmax_margin", fusion_temp=2.0, manifest_sha256="dummy_hash")
        match, reason = validate_protocol_signature(sig1, sig2)
        self.assertFalse(match)
        self.assertIn("fusion_temp", reason)

    # 16. Exhaustive manifest validation and error detection (P0.2)
    def test_16_validate_manifest_exhaustive_error_detection(self):
        # A. Missing episode_id
        bad_path = os.path.join(self.episodes_dir, "missing_epid.json")
        bad_data = {
            "seed": 42, "benchmark": "isic",
            "episodes": [
                {"query_img": "q.jpg", "query_mask": "q.png", "support_imgs": ["s.jpg"], "support_masks": ["s.png"]}
            ]
        }
        with open(bad_path, "w") as f:
            json.dump(bad_data, f)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(bad_path, benchmark="isic")
        self.assertIn("Missing episode_id", str(ctx.exception))

        # B. Duplicate episode_id
        dup_path = os.path.join(self.episodes_dir, "dup_epid.json")
        dup_data = {
            "seed": 42, "benchmark": "isic",
            "episodes": [
                {"episode_id": 1, "query_img": "q1.jpg", "query_mask": "q1.png", "support_imgs": ["s1.jpg"], "support_masks": ["s1.png"]},
                {"episode_id": 1, "query_img": "q2.jpg", "query_mask": "q2.png", "support_imgs": ["s2.jpg"], "support_masks": ["s2.png"]}
            ]
        }
        with open(dup_path, "w") as f:
            json.dump(dup_data, f)
        with self.assertRaises(ValueError) as ctx2:
            validate_manifest(dup_path, benchmark="isic")
        self.assertIn("Duplicate episode_id", str(ctx2.exception))

        # C. Non-existent file references under dataset_base_path
        nonexistent_path = os.path.join(self.episodes_dir, "nonexistent_files.json")
        ne_data = {
            "seed": 42, "benchmark": "isic", "nshot": 1,
            "episodes": [
                {"episode_id": 0, "query_img": "nonexistent/q.jpg", "query_mask": "nonexistent/q.png", "support_imgs": ["nonexistent/s.jpg"], "support_masks": ["nonexistent/s.png"]}
            ]
        }
        with open(nonexistent_path, "w") as f:
            json.dump(ne_data, f)
        fake_ds_dir = os.path.join(self.temp_dir.name, "fake_ds")
        os.makedirs(fake_ds_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError) as ctx3:
            validate_manifest(nonexistent_path, benchmark="isic", dataset_base_path=fake_ds_dir)
        self.assertIn("does not exist", str(ctx3.exception))

        # D. Support set size != nshot
        nshot_mismatch_path = os.path.join(self.episodes_dir, "nshot_mismatch.json")
        ns_data = {
            "seed": 42, "benchmark": "isic", "nshot": 1,
            "episodes": [
                {"episode_id": 0, "query_img": "q.jpg", "query_mask": "q.png", "support_imgs": ["s1.jpg", "s2.jpg"], "support_masks": ["s1.png", "s2.png"]}
            ]
        }
        with open(nshot_mismatch_path, "w") as f:
            json.dump(ns_data, f)
        with self.assertRaises(ValueError) as ctx4:
            validate_manifest(nshot_mismatch_path, benchmark="isic", nshot=1)
        self.assertIn("support set size", str(ctx4.exception))

    # 17. check_reproducibility compares by episode_id and checks pre-validation (P1.5)
    def test_17_check_reproducibility_by_episode_id_and_prevalidation(self):
        from experiments.check_reproducibility import compare_runs

        # Base run artifact with detailed_episodes
        sig = create_protocol_signature("lung", "E0", 3, seed=42, manifest_sha256="valid_sha256")
        run1 = {
            "protocol_signature": sig,
            "episode_count": 3,
            "metric": {"Mean_Episode_IoU": 60.0, "Cumulative_mIoU": 60.0},
            "raw_result": {
                "episode_ious": [60.0, 60.1, 59.9],
                "detailed_episodes": [
                    {"episode_id": 0, "iou": 60.0},
                    {"episode_id": 1, "iou": 60.1},
                    {"episode_id": 2, "iou": 59.9}
                ]
            }
        }
        # Run 2: same episodes but intentionally permuted in list order
        run2_permuted = {
            "protocol_signature": sig,
            "episode_count": 3,
            "metric": {"Mean_Episode_IoU": 60.02, "Cumulative_mIoU": 60.02},
            "raw_result": {
                "episode_ious": [59.95, 60.02, 60.09],
                "detailed_episodes": [
                    {"episode_id": 2, "iou": 59.95},
                    {"episode_id": 0, "iou": 60.02},
                    {"episode_id": 1, "iou": 60.09}
                ]
            }
        }
        r1_path = os.path.join(self.temp_dir.name, "run1.json")
        r2_path = os.path.join(self.temp_dir.name, "run2.json")
        with open(r1_path, "w") as f: json.dump(run1, f)
        with open(r2_path, "w") as f: json.dump(run2_permuted, f)

        # Comparison aligned by episode_id should PASS despite permutation
        res = compare_runs(r1_path, r2_path)
        self.assertTrue(res["overall_pass"])
        self.assertTrue(res["manifest_match"])
        self.assertTrue(res["count_match"])
        self.assertTrue(res["ids_match"])
        self.assertLess(res["max_episode_delta"], 0.1)

        # Mismatched episode count must FAIL
        run3_mismatch = dict(run1)
        run3_mismatch["episode_count"] = 2
        run3_mismatch["raw_result"] = {
            "episode_ious": [60.0, 60.1],
            "detailed_episodes": [{"episode_id": 0, "iou": 60.0}, {"episode_id": 1, "iou": 60.1}]
        }
        r3_path = os.path.join(self.temp_dir.name, "run3.json")
        with open(r3_path, "w") as f: json.dump(run3_mismatch, f)
        res_fail_count = compare_runs(r1_path, r3_path)
        self.assertFalse(res_fail_count["overall_pass"])
        self.assertFalse(res_fail_count["count_match"])

        # Mismatched manifest SHA256 must FAIL
        run4_bad_sha = dict(run1)
        run4_bad_sha["protocol_signature"] = create_protocol_signature("lung", "E0", 3, seed=42, manifest_sha256="different_sha")
        r4_path = os.path.join(self.temp_dir.name, "run4.json")
        with open(r4_path, "w") as f: json.dump(run4_bad_sha, f)
        res_fail_sha = compare_runs(r1_path, r4_path)
        self.assertFalse(res_fail_sha["overall_pass"])
        self.assertFalse(res_fail_sha["manifest_match"])

    # 18. Resume validation rejects when signature, manifest hash, or episode count differ (P1.6)
    def test_18_resume_validation_rules(self):
        sig_canonical = create_protocol_signature("isic", "E0", 100, seed=42, manifest_sha256="canonical_hash_123")
        
        # Mismatched fusion_temp
        sig_diff_temp = create_protocol_signature("isic", "E0", 100, seed=42, fusion_temp=2.0, manifest_sha256="canonical_hash_123")
        match, reason = validate_protocol_signature(sig_diff_temp, sig_canonical)
        self.assertFalse(match)

        # Mismatched manifest_sha256
        sig_diff_hash = create_protocol_signature("isic", "E0", 100, seed=42, manifest_sha256="tampered_hash_456")
        match, reason = validate_protocol_signature(sig_diff_hash, sig_canonical)
        self.assertFalse(match)

        # Mismatched episode count
        sig_diff_count = create_protocol_signature("isic", "E0", 50, seed=42, manifest_sha256="canonical_hash_123")
        match, reason = validate_protocol_signature(sig_diff_count, sig_canonical)
        self.assertFalse(match)

    # 19. Exhaustive --episodes all generation across all 5 datasets (P0.1 & P0.3)
    def test_19_exhaustive_all_manifest_generation_all_5_datasets(self):
        from experiments.generate_manifests import (
            SAMPLERS,
            check_or_download_dataset,
            KAGGLE_SLUGS
        )
        from src.datasets.deepglobe import DeepglobeDataset
        from src.datasets.isic import ISICDataset
        from src.datasets.lung import LungDataset
        from src.datasets.fss import FSS1000Dataset
        from src.datasets.suim import SUIMDataset

        loaders = {
            'deepglobe': (DeepglobeDataset, 1833),
            'isic': (ISICDataset, 2594),
            'lung': (LungDataset, 704),
            'fss': (FSS1000Dataset, 2400),
            'suim': (SUIMDataset, 3859)
        }

        for b_name, (ds_cls, expected_count) in loaders.items():
            datapath = check_or_download_dataset(b_name, KAGGLE_SLUGS.get(b_name))
            self.assertTrue(os.path.exists(datapath), f"Dataset path missing for {b_name}")

            sampler = SAMPLERS[b_name]
            episodes = sampler(datapath, num_episodes='all', seed=42, shot=1)

            # 1. Total count matches exhaustive dataset count
            self.assertEqual(len(episodes), expected_count, f"{b_name} episode count mismatch: {len(episodes)} vs {expected_count}")

            # 2. Episode IDs strictly unique and continuous 0..N-1
            ep_ids = [e['episode_id'] for e in episodes]
            self.assertEqual(len(ep_ids), len(set(ep_ids)), f"{b_name} contains duplicate episode_id!")
            self.assertEqual(ep_ids, list(range(len(episodes))), f"{b_name} episode_ids are not continuous 0..N-1")

            # 3. Query coverage: 100% queries, 0 duplicate, 0 missing
            if b_name == 'suim':
                # In SUIM, queries are category-specific masks across 7 classes
                queries = [e['query_mask'] for e in episodes]
            else:
                queries = [e['query_img'] for e in episodes]

            self.assertEqual(len(queries), len(set(queries)), f"{b_name} has duplicate query instances in exhaustive mode!")
            self.assertEqual(len(queries), expected_count, f"{b_name} missing query instances!")

            # 4. Support sets do not contain query image
            for ep in episodes:
                self.assertNotIn(ep['query_img'], ep['support_imgs'], f"{b_name} episode {ep['episode_id']} support contains query image!")
                self.assertEqual(len(ep['support_imgs']), 1, f"{b_name} 1-shot support set size mismatch")

    # 20. E0–E3 shared manifest SHA256 across all 5 datasets (P0.3)
    def test_20_shared_manifest_sha256_across_e0_e3_all_datasets(self):
        from experiments.generate_manifests import generate_manifest, check_or_download_dataset, KAGGLE_SLUGS

        for b_name in ['deepglobe', 'isic', 'lung', 'fss', 'suim']:
            datapath = check_or_download_dataset(b_name, KAGGLE_SLUGS.get(b_name))
            m_path = generate_manifest(
                benchmark=b_name,
                episodes=20,
                seed=42,
                shot=1,
                datapath=datapath,
                out_file=os.path.join(self.episodes_dir, f"{b_name}_seed42_20episodes.json")
            )
            val = validate_manifest(m_path, benchmark=b_name, seed=42, episodes=20, dataset_base_path=datapath)
            sha_original = val['manifest_sha256']

            # E0-E3 all resolve and verify the identical manifest SHA256
            for exp in ['E0', 'E1', 'E2', 'E3']:
                resolved = resolve_manifest_path(b_name, seed=42, episodes=20, base_dir=self.episodes_dir)
                resolved_val = validate_manifest(resolved, benchmark=b_name, seed=42, episodes=20)
                self.assertEqual(resolved_val['manifest_sha256'], sha_original, f"{b_name} {exp} manifest SHA256 deviated")

if __name__ == '__main__':
    unittest.main()

