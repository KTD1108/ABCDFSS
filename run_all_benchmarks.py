#!/usr/bin/env python3
"""
Automated Master Runner for All 5 CD-FSS Benchmarks
Runs full evaluation across all domains, saves detailed individual logs,
and outputs a comprehensive verified benchmark summary table.

Usage:
    python run_all_benchmarks.py
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def check_or_download_dataset(benchmark: str, kaggle_slug: str = None) -> str:
    """Tự động tải trực tiếp từ internet nếu chưa có trong cache/thư mục cục bộ."""
    # 1. Kiểm tra cache Kaggle
    if kaggle_slug:
        candidates = [
            f"/root/.cache/kagglehub/datasets/{kaggle_slug}/versions/7",
            f"/root/.cache/kagglehub/datasets/{kaggle_slug}/versions/2",
            f"/root/.cache/kagglehub/datasets/{kaggle_slug}/versions/1",
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"[✓] Đã có sẵn '{benchmark}' trong cache: {c}")
                return c

    # 2. Xử lý chuyên biệt cho FSS-1000
    if benchmark == 'fss':
        # Kiểm tra xem đã có sẵn ảnh trong cache Kaggle hoặc cục bộ chưa
        fss_check_dirs = [
            "/root/.cache/kagglehub/datasets/nikhilpandey360/fss-1000-a-1000-class-few-shot-segmentation/versions/1",
            "/root/.cache/kagglehub/datasets/itsahmad/fss-1000/versions/1",
            "./datasets/fss1000",
            "./datasets/fss"
        ]
        for cd in fss_check_dirs:
            if os.path.exists(cd):
                for root, dirs, files in os.walk(cd):
                    if any(f.endswith('.jpg') for f in files) or any(d in ['bus', 'pizza', 'spiderman'] for d in dirs):
                        print(f"[✓] Đã có sẵn FSS-1000 với đầy đủ ảnh tại: {root}")
                        return root

        print("[*] Đang tự động tải FSS-1000...")
        # Ưu tiên 1: Tải trực tiếp qua kagglehub (ổn định, không bị chặn link)
        kaggle_fss_slugs = [
            "nikhilpandey360/fss-1000-a-1000-class-few-shot-segmentation",
            "itsahmad/fss-1000"
        ]
        for slug in kaggle_fss_slugs:
            try:
                print(f"[*] Đang tải FSS-1000 từ Kaggle ({slug})...")
                import kagglehub
                p = kagglehub.dataset_download(slug)
                print(f"[✓] Tải thành công FSS-1000 từ Kaggle về: {p}")
                return p
            except Exception as ke:
                print(f"[-] Không thể tải từ Kaggle ({slug}): {ke}")

        # Ưu tiên 2: Tải từ Google Drive qua gdown
        os.makedirs("./datasets", exist_ok=True)
        zip_path = "./datasets/fss1000.zip"
        dest_dir = "./datasets/fss1000"
        gdrive_ids = ["16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI", "1tt3dkdASjXt58t-2A9zeucZ397ZRF7In"]

        for gid in gdrive_ids:
            try:
                import gdown, zipfile
                if os.path.exists(zip_path) and os.path.getsize(zip_path) < 10000000:
                    os.remove(zip_path)
                if not os.path.exists(zip_path):
                    print(f"[*] Đang tải fss1000.zip từ Google Drive ID ({gid})...")
                    gdown.download(id=gid, output=zip_path, quiet=False, fuzzy=True)
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 10000000:
                    print("[*] Đang giải nén FSS-1000...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dest_dir)
                    print(f"[✓] Tải và giải nén FSS-1000 thành công vào: {dest_dir}")
                    return dest_dir
            except Exception as gde:
                print(f"[-] Lỗi với Google Drive ID {gid}: {gde}")

        return dest_dir

    # 3. Kiểm tra thư mục cục bộ ./datasets
    local_candidates = [
        f"./datasets/{benchmark}",
        f"./datasets/{benchmark}1000"
    ]
    for lc in local_candidates:
        if os.path.exists(lc) and len(os.listdir(lc)) > 0:
            print(f"[✓] Đã có sẵn '{benchmark}' tại thư mục cục bộ: {lc}")
            return lc

    # 4. Tải tự động từ Kaggle qua kagglehub
    if kaggle_slug:
        print(f"[*] Đang tự động tải '{benchmark}' từ Kaggle ({kaggle_slug})...")
        try:
            import kagglehub
            path = kagglehub.dataset_download(kaggle_slug)
            print(f"[✓] Tải thành công '{benchmark}' về: {path}")
            return path
        except Exception as e:
            print(f"[!] Lỗi khi tải {benchmark} từ kagglehub: {e}")

    return f"./datasets/{benchmark}"

def main():
    print("=" * 80)
    print("      QUY TRÌNH CHẠY TỰ ĐỘNG TOÀN BỘ 5 BỘ DỮ LIỆU CD-FSS (1-SHOT, NO-PP)")
    print("=" * 80)

    # 1. Xác định đường dẫn cho cả 5 bộ dữ liệu
    dataset_paths = {
        'isic': check_or_download_dataset('isic', kaggle_slug='heyoujue/isic2018-classwise'),
        'suim': check_or_download_dataset('suim', kaggle_slug='heyoujue/suim-merged'),
        'lung': check_or_download_dataset('lung', kaggle_slug='heyoujue/lungsegmentation'),
        'fss': check_or_download_dataset('fss'),
        'deepglobe': check_or_download_dataset('deepglobe', kaggle_slug='heyoujue/deepglobe')
    }

    # 2. Danh sách các thử nghiệm cần chạy
    experiments = [
        # Domain 1: Natural Objects
        {
            'name': 'FSS-1000 (Cải tiến Đề xuất)',
            'benchmark': 'fss',
            'datapath': dataset_paths['fss'],
            'adapter': 'depthwise_separable_3x3',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        # Domain 2: Dermatology
        {
            'name': 'ISIC (Cải tiến Fusion)',
            'benchmark': 'isic',
            'datapath': dataset_paths['isic'],
            'adapter': 'conv1x1',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        {
            'name': 'ISIC (Cải tiến Adapter)',
            'benchmark': 'isic',
            'datapath': dataset_paths['isic'],
            'adapter': 'depthwise_separable_3x3',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        # Domain 3: Underwater
        {
            'name': 'SUIM (Cải tiến Fusion)',
            'benchmark': 'suim',
            'datapath': dataset_paths['suim'],
            'adapter': 'conv1x1',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        {
            'name': 'SUIM (Cải tiến Adapter)',
            'benchmark': 'suim',
            'datapath': dataset_paths['suim'],
            'adapter': 'depthwise_separable_3x3',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        # Domain 4: Radiology
        {
            'name': 'Lung / CXR (Cải tiến Đề xuất)',
            'benchmark': 'lung',
            'datapath': dataset_paths['lung'],
            'adapter': 'depthwise_separable_3x3',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
        # Domain 5: Satellite
        {
            'name': 'Deepglobe (Cải tiến Đề xuất)',
            'benchmark': 'deepglobe',
            'datapath': dataset_paths['deepglobe'],
            'adapter': 'depthwise_separable_3x3',
            'fusion': 'softmax_margin',
            'nshot': 1
        },
    ]

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    summary_results = []

    # 3. Lần lượt chạy từng thử nghiệm
    total_exp = len(experiments)
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*30} TIẾN TRÌNH [{i}/{total_exp}]: {exp['name']} {'='*30}")
        if not os.path.exists(exp['datapath']):
            print(f"[!] CẢNH BÁO: Không tìm thấy thư mục dữ liệu {exp['datapath']}. Bỏ qua...")
            continue

        cmd = [
            sys.executable, "evaluate.py",
            "--benchmark", exp['benchmark'],
            "--datapath", exp['datapath'],
            "--nshot", str(exp['nshot']),
            "--adapter", exp['adapter'],
            "--fusion", exp['fusion'],
            "--logpath", log_dir,
            "--device", "cuda"
        ]

        print(f"[*] Thực thi lệnh: {' '.join(cmd)}")
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = project_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
        result = subprocess.run(cmd, cwd=project_root, env=env)
        if result.returncode != 0:
            print(f"[!] Lỗi khi chạy thử nghiệm {exp['name']}")

    # 4. Đọc lại file summary_records.jsonl và tạo bảng Markdown tổng kết
    summary_jsonl = os.path.join(log_dir, "summary_records.jsonl")
    if os.path.exists(summary_jsonl):
        records = []
        with open(summary_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        print("\n\n" + "=" * 80)
        print("           BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM ĐÃ XÁC THỰC TỪ LOG")
        print("=" * 80)
        md_table = [
            "| Thời gian | Benchmark | Shot | Adapter | Fusion | mIoU (%) | FB-IoU (%) | File Log Chi Tiết |",
            "| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |"
        ]
        for r in records:
            md_table.append(f"| {r['timestamp']} | **{r['benchmark'].upper()}** | {r['nshot']} | `{r['adapter']}` | `{r['fusion']}` | **{r['mIoU']}%** | **{r['FB-IoU']}%** | `{os.path.basename(r['log_file'])}` |")

        table_str = "\n".join(md_table)
        print(table_str)

        output_md_file = os.path.join(log_dir, "verified_benchmark_summary.md")
        with open(output_md_file, "w", encoding="utf-8") as f:
            f.write("# BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐÃ KIỂM CHỨNG TỪ LOG\n\n" + table_str + "\n")
        print(f"\n[✓] Đã xuất file tóm tắt ra: {output_md_file}")

if __name__ == '__main__':
    main()
