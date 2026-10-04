#!/usr/bin/env python3
"""
Tự động dọn dẹp thư mục logs:
- Chỉ giữ lại đúng 10 file log chuẩn của lượt chạy hoàn chỉnh mới nhất
- Giữ lại 2 file tổng kết: verified_benchmark_summary.md và summary_records.jsonl
- Xóa toàn bộ các file log lỗi hoặc chạy dở từ các lượt chạy trước
- Tự động đóng gói thành file OFFICIAL_10_LOGS.zip để tải về máy
"""

import os
import glob
import json
import zipfile

OFFICIAL_LOG_FILES = [
    "fss_conv1x1_softmax_margin_1shot_20261004_094041.log",
    "fss_depthwise_separable_3x3_softmax_margin_1shot_20261004_095507.log",
    "isic_conv1x1_softmax_margin_1shot_20261004_101331.log",
    "isic_depthwise_separable_3x3_softmax_margin_1shot_20261004_101608.log",
    "suim_conv1x1_softmax_margin_1shot_20261004_101901.log",
    "suim_depthwise_separable_3x3_softmax_margin_1shot_20261004_102246.log",
    "lung_conv1x1_softmax_margin_1shot_20261004_102656.log",
    "lung_depthwise_separable_3x3_softmax_margin_1shot_20261004_102804.log",
    "deepglobe_conv1x1_softmax_margin_1shot_20261004_102914.log",
    "deepglobe_depthwise_separable_3x3_softmax_margin_1shot_20261004_103108.log",
    "verified_benchmark_summary.md",
    "summary_records.jsonl"
]

def clean_logs():
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        print("[!] Không tìm thấy thư mục ./logs")
        return

    # 1. Lọc và chuẩn hóa file summary_records.jsonl
    summary_jsonl = os.path.join(log_dir, "summary_records.jsonl")
    if os.path.exists(summary_jsonl):
        official_records = []
        with open(summary_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line.strip())
                        log_base = os.path.basename(rec.get('log_file', ''))
                        if log_base in OFFICIAL_LOG_FILES:
                            official_records.append(rec)
                    except Exception:
                        pass
        with open(summary_jsonl, 'w', encoding='utf-8') as f:
            for rec in official_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"[✓] Đã chuẩn hóa summary_records.jsonl (còn đúng {len(official_records)} bản ghi chính thức).")

    # 2. Xóa tất cả các file log cũ bị lỗi hoặc chạy dở
    all_files = glob.glob(os.path.join(log_dir, "*"))
    deleted_count = 0
    retained_count = 0

    for fpath in all_files:
        basename = os.path.basename(fpath)
        if basename.endswith(".zip"):
            continue
        if basename not in OFFICIAL_LOG_FILES:
            try:
                os.remove(fpath)
                deleted_count += 1
            except Exception as e:
                print(f"[-] Không thể xóa {basename}: {e}")
        else:
            retained_count += 1

    print(f"[✓] Đã xóa {deleted_count} file log rác/chạy dở.")
    print(f"[✓] Đã giữ lại đúng {retained_count} file log chuẩn xác nhất:")
    for f in OFFICIAL_LOG_FILES:
        if os.path.exists(os.path.join(log_dir, f)):
            print(f"    - {f}")

    # 3. Nén thành file zip chuẩn duy nhất
    zip_output = os.path.join(log_dir, "OFFICIAL_10_LOGS.zip")
    with zipfile.ZipFile(zip_output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in OFFICIAL_LOG_FILES:
            p = os.path.join(log_dir, f)
            if os.path.exists(p):
                zf.write(p, arcname=f)

    print(f"\n[✓] Đã đóng gói thành công file nén chuẩn: {zip_output}")
    print(f"[*] Kích thước file zip: {round(os.path.getsize(zip_output) / 1024, 2)} KB")

if __name__ == '__main__':
    clean_logs()
