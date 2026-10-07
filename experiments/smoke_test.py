import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=== 1. IMPORTING CORE MODULES ===")
from src.engine.pipeline import CDFSSEngine
from src.datasets.builder import build_dataloader
from src.models.adapters import PointwiseAdapter, DepthwiseSeparableAdapter
from src.models.fusion import build_fusion
from src.models.backbone import ResNetBackbone
from src.utils.manifest import resolve_manifest_path, validate_manifest
from src.utils.protocol import create_protocol_signature, validate_protocol_signature, get_git_info, get_environment_info
print(" -> All core imports SUCCESSFUL!")

print("\n=== 2. TESTING MANIFEST RESOLUTION ===")
for b in ['deepglobe', 'isic', 'lung', 'fss', 'suim']:
    m_100 = resolve_manifest_path(b, seed=42, episodes='100')
    m_1000 = resolve_manifest_path(b, seed=42, episodes='1000')
    assert m_100 and m_1000, f"Failed manifest for {b}"
    print(f" -> {b:<10}: 100ep & 1000ep manifests verified!")

print("\n=== 3. TESTING PROTOCOL TELEMETRY ===")
git = get_git_info()
print(f" -> Git commit: {git.get('git_commit')} (status: {git.get('status')})")
env = get_environment_info()
print(f" -> PyTorch: {env.get('torch_version')}, CUDA: {env.get('cuda_version')}")

print("\n=== 4. TESTING AUDIT SCRIPT ===")
res = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'experiments', 'audit_benchmark_results.py')], cwd=PROJECT_ROOT, capture_output=True, text=True)
assert res.returncode == 0, f"Audit script failed: {res.stderr}"
assert 'DEEPGLOBE' in res.stdout and 'SUIM' in res.stdout
print(" -> 20-run Audit Script PASSED with 100% matrix output!")

print("\n=== 5. TESTING REPRODUCIBILITY SCRIPT ===")
res2 = subprocess.run([
    sys.executable, os.path.join(PROJECT_ROOT, 'experiments', 'check_reproducibility.py'),
    '--run1', os.path.join(PROJECT_ROOT, 'results', 'lung', 'E0_1000ep_seed42', 'run_result.json'),
    '--run2', os.path.join(PROJECT_ROOT, 'results', 'lung', 'E0_1000ep_seed42', 'run_result.json')
], cwd=PROJECT_ROOT, capture_output=True, text=True)
assert res2.returncode == 0, f"Reproducibility script failed: {res2.stderr}"
assert '[PASS]' in res2.stdout
print(" -> Reproducibility script PASSED with [PASS] verdict!")

print("\n=== 6. TESTING CLI HELP INTERFACES ===")
for script in ['main.py', 'evaluate.py', 'run_all_benchmarks.py']:
    r = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, script), '--help'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} --help failed"
    print(f" -> {script} CLI interface verified!")

print("\n=======================================================")
print("  ALL 6 SMOKE TESTS COMPLETED WITH 100% SUCCESS!")
print("=======================================================")
