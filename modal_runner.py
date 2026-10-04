#!/usr/bin/env python3
"""
Modal Serverless GPU Runner for CD-FSS Benchmarking
Runs evaluation directly on cloud GPU (T4 / A10G) matching the CVPR 2024 exact protocol:
  Algorithm 2: --adapt-to every-episode --episodes 1000

Usage examples:
  # 1. Chạy 1 benchmark cụ thể (ví dụ: DeepGlobe hoặc Lung)
  modal run modal_runner.py --benchmark deepglobe --adapt-to every-episode --episodes 1000 --adapter depthwise_separable_3x3 --fusion softmax_margin

  # 2. Tái lập thuần bài báo gốc (Pure CVPR 2024 Baseline)
  modal run modal_runner.py --benchmark deepglobe --adapt-to every-episode --episodes 1000 --adapter conv1x1 --fusion mean
"""

import os
import subprocess
import modal

# 1. Khởi tạo Modal App
app = modal.App("abcdfss-cd-fss")

# 2. Định nghĩa Docker Image với CUDA và các thư viện cần thiết
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "unzip", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python-headless",
        "numpy<2.0.0",
        "Pillow",
        "scipy",
        "tqdm",
        "kagglehub",
    )
    .add_local_dir(
        LOCAL_DIR,
        remote_path="/root/ABCDFSS",
        ignore=[".git", "*__pycache__*", "*.pdf", "*.log", "logs/*"]
    )
)

@app.function(
    image=image,
    gpu="T4",               # Mặc định GPU T4 ($0.59/h) hoặc A10G
    timeout=7200,          # 2 giờ tối đa
)
def run_evaluation(
    benchmark: str = "deepglobe",
    nshot: int = 1,
    adapter: str = "depthwise_separable_3x3",
    fusion: str = "softmax_margin",
    fusion_temp: float = 1.0,
    adapt_to: str = "every-episode",
    episodes: int = 1000,
    postprocessing: str = "off",
):
    import torch
    print("=" * 70)
    print("      ABCDFSS GPU EVALUATION ON MODAL CLOUD (CVPR 2024 PROTOCOL)")
    print(f"Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Benchmark: {benchmark.upper()} | {nshot}-shot")
    print(f"Adapter:   {adapter} | Fusion: {fusion} (temp={fusion_temp})")
    print(f"Mode:      {adapt_to} (Algorithm 2: Every-Episode)")
    print(f"Episodes:  {episodes} episodes")
    print("=" * 70)

    # 1. Tự động tải dữ liệu nếu chưa có
    from run_all_benchmarks import check_or_download_dataset
    kaggle_slugs = {
        'isic': 'heyoujue/isic2018-classwise',
        'suim': 'heyoujue/suim-merged',
        'lung': 'heyoujue/lungsegmentation',
        'deepglobe': 'heyoujue/deepglobe'
    }
    datapath = check_or_download_dataset(benchmark, kaggle_slug=kaggle_slugs.get(benchmark))

    # 2. Xây dựng lệnh evaluate.py
    cmd = [
        "python", "evaluate.py",
        "--benchmark", benchmark,
        "--datapath", datapath,
        "--nshot", str(nshot),
        "--adapter", adapter,
        "--fusion", fusion,
        "--fusion-temp", str(fusion_temp),
        "--adapt-to", adapt_to,
        "--episodes", str(episodes),
        "--postprocessing", postprocessing,
        "--device", "cuda"
    ]

    print(f"\n[*] Thực thi lệnh: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ABCDFSS"

    process = subprocess.Popen(
        cmd,
        cwd="/root/ABCDFSS",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)

    process.wait()
    print("=" * 70)
    print(f"[*] Hoàn thành thử nghiệm với exit code: {process.returncode}")
    print("=" * 70)

@app.local_entrypoint()
def main(
    benchmark: str = "deepglobe",
    nshot: int = 1,
    adapter: str = "depthwise_separable_3x3",
    fusion: str = "softmax_margin",
    fusion_temp: float = 1.0,
    adapt_to: str = "every-episode",
    episodes: int = 1000,
    postprocessing: str = "off",
):
    run_evaluation.remote(
        benchmark=benchmark,
        nshot=nshot,
        adapter=adapter,
        fusion=fusion,
        fusion_temp=fusion_temp,
        adapt_to=adapt_to,
        episodes=episodes,
        postprocessing=postprocessing
    )
