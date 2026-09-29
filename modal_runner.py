r"""
Modal Serverless GPU Runner for ABCDFSS (Cross-Domain Few-Shot Segmentation)
Run evaluation on cloud GPUs (T4 / A10G / L4) with data mounted from persistent Modal Volume.
"""
import os
import sys
import shutil
import subprocess
import modal

# 1. Khởi tạo Modal App
app = modal.App("abcdfss-cd-fss")

# 2. Định nghĩa Modal Persistent Volume để lưu trữ vĩnh viễn dataset (FSS-1000, Deepglobe, etc.)
volume = modal.Volume.from_name("abcdfss-data", create_if_missing=True)

# 3. Định nghĩa môi trường Docker Image với PyTorch CUDA 12.1 và các thư viện cần thiết
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
        "tensorboardX",
        "gdown",
    )
    .add_local_dir(
        LOCAL_DIR,
        remote_path="/root/ABCDFSS",
        ignore=[".git", "*__pycache__*", "*.pdf", "*.log", "logs/*"]
    )
)


DATASET_DRIVE_IDS = {
    "deepglobe": "1ktPxbmkNzNsZuEO_sUvyt27T-w8xFbzd",
    "fss1000": "1tt3dkdASjXt58t-2A9zeucZ397ZRF7In",
}

# ---------------------------------------------------------------------------
# FUNCTION 1: Tải dữ liệu từ Google Drive vào Modal Volume (chỉ chạy 1 lần)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2 giờ cho các file lớn
)
def sync_from_gdrive(target_name: str = "deepglobe", drive_id: str = "", is_folder: bool = False):
    r"""
    Tải file zip hoặc folder từ Google Drive trực tiếp vào Modal Volume /data/{target_name}
    Ví dụ:
      modal run modal_runner.py::sync_from_gdrive --target-name deepglobe
      modal run modal_runner.py::sync_from_gdrive --target-name fss1000
    """
    import gdown

    if not drive_id:
        drive_id = DATASET_DRIVE_IDS.get(target_name, "")
    if not drive_id:
        raise ValueError(f"Không tìm thấy Google Drive ID cho '{target_name}'. Vui lòng truyền --drive-id.")

    os.makedirs(f"/data/{target_name}", exist_ok=True)
    temp_zip = f"/tmp/{target_name}.zip"

    print(f"[*] Bắt đầu tải dữ liệu cho '{target_name}' từ Google Drive ID: {drive_id}...")
    
    if is_folder:
        url = f"https://drive.google.com/drive/folders/{drive_id}"
        gdown.download_folder(url, output=f"/data/{target_name}", quiet=False, use_cookies=False)
    else:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, temp_zip, quiet=False)

        if os.path.exists(temp_zip):
            file_size_mb = os.path.getsize(temp_zip) / (1024 * 1024)
            print(f"[+] Tải thành công ({file_size_mb:.2f} MB). Đang giải nén vào /data/{target_name}...")
            shutil.unpack_archive(temp_zip, f"/data/{target_name}")
            os.remove(temp_zip)
            print("[+] Giải nén hoàn tất!")
        else:
            print("[!] Không tìm thấy file đã tải. Vui lòng kiểm tra lại Google Drive ID và quyền chia sẻ công khai.")

    # Commit volume để lưu dữ liệu vĩnh viễn trên Modal Cloud
    volume.commit()
    print(f"[+] Dữ liệu '{target_name}' đã được lưu vĩnh viễn trên Modal Volume 'abcdfss-data'!")


# ---------------------------------------------------------------------------
# FUNCTION 2: Kiểm tra danh mục thư mục & file trên Modal Volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def check_volume():
    r""" Kiểm tra nội dung trong /data trên Modal Volume """
    print("=== NỘI DUNG MODAL VOLUME (/data) ===")
    if not os.path.exists("/data") or not os.listdir("/data"):
        print("Volume trống. Vui lòng tải dữ liệu bằng sync_from_gdrive.")
        return

    for root, dirs, files in os.walk("/data"):
        level = root.replace("/data", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")
        if level > 2:
            dirs.clear()  # Giới hạn độ sâu in


# ---------------------------------------------------------------------------
# FUNCTION 3: Chạy Đánh Giá Phân Đoạn Trên GPU (T4 / A10G / L4)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",               # Mặc định GPU T4 ($0.59/h). Có thể đổi sang "A10G" hoặc "L4"
    volumes={"/data": volume},
    timeout=7200,          # 2 giờ
)
def evaluate(
    benchmark: str = "deepglobe",
    nshot: int = 1,
    adapter_type: str = "depthwise_separable_3x3",
    fusion_mode: str = "softmax_margin",
    fusion_temp: float = 1.0,
    adapt_to: str = "first-episode",
    postprocessing: str = "off",
    verbosity: int = 1
):
    r"""
    Chạy thử nghiệm CD-FSS trên GPU Modal.
    Ví dụ:
      modal run modal_runner.py::evaluate --benchmark deepglobe --nshot 1 --adapter-type depthwise_separable_3x3 --fusion-mode softmax_margin
    """
    import torch
    print("=" * 60)
    print("ABCDFSS GPU EVALUATION ON MODAL CLOUD")
    print(f"Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Benchmark: {benchmark.upper()} | {nshot}-shot")
    print(f"Adapter: {adapter_type} | Fusion: {fusion_mode} (temp={fusion_temp})")
    print(f"Mode: {adapt_to} | Post-processing: {postprocessing}")
    print("=" * 60)

    # Xác định đường dẫn datapath phù hợp với từng benchmark trên /data
    if benchmark == "deepglobe":
        datapath = "/data/deepglobe"
        # Xử lý trường hợp giải nén tạo thư mục lồng
        if os.path.exists("/data/deepglobe/deepglobe"):
            datapath = "/data/deepglobe/deepglobe"
    elif benchmark in ["fss", "fss1000"]:
        benchmark = "fss"
        datapath = "/data/fss1000"
        if os.path.exists("/data/fss1000/FSS-1000"):
            datapath = "/data/fss1000"
    else:
        datapath = f"/data/{benchmark}"

    print(f"[*] Sử dụng datapath: {datapath}")
    if not os.path.exists(datapath):
        print(f"[!] LỖI: Không tìm thấy thư mục {datapath} trên Volume.")
        print("[!] Vui lòng chạy `modal run modal_runner.py::check_volume` để xem các thư mục hiện có.")
        return

    # Chuẩn bị lệnh gọi main.py
    cmd = [
        "python", "main.py",
        "--benchmark", benchmark,
        "--datapath", datapath,
        "--nshot", str(nshot),
        "--adapt-to", adapt_to,
        "--postprocessing", postprocessing,
        "--adapter-type", adapter_type,
        "--fusion-mode", fusion_mode,
        "--fusion-temp", str(fusion_temp),
        "--verbosity", str(verbosity),
    ]

    print(f"[*] Thực thi: {' '.join(cmd)}")
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
    print("=" * 60)
    print(f"[*] Hoàn thành thử nghiệm với exit code: {process.returncode}")
    print("=" * 60)
