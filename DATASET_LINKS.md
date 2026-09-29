# DANH SÁCH LIÊN KẾT DỮ LIỆU GOOGLE DRIVE (DATASET GOOGLE DRIVE LINKS)

Tài liệu này lưu trữ đường link và File ID của các tập dữ liệu phục vụ nghiên cứu và chạy thực nghiệm CD-FSS trên Modal / Colab / Kaggle.

---

## 1. FSS-1000 (Few-Shot Semantic Segmentation 1000 classes)
- **Đường link đầy đủ**: 
  `https://drive.google.com/file/d/1tt3dkdASjXt58t-2A9zeucZ397ZRF7In/view?usp=sharing`
- **Google Drive File ID**: 
  `1tt3dkdASjXt58t-2A9zeucZ397ZRF7In`
- **Lệnh tải tự động với gdown**:
  ```python
  import gdown
  gdown.download(id="1tt3dkdASjXt58t-2A9zeucZ397ZRF7In", output="datasets/fss1000.zip", quiet=False)
  ```

---

## 2. Deepglobe (Land Cover Remote Sensing)
- **Đường link đầy đủ**: 
  `https://drive.google.com/file/d/1ktPxbmkNzNsZuEO_sUvyt27T-w8xFbzd/view?usp=sharing`
- **Google Drive File ID**: 
  `1ktPxbmkNzNsZuEO_sUvyt27T-w8xFbzd`
- **Lệnh tải tự động với gdown**:
  ```python
  import gdown
  gdown.download(id="1ktPxbmkNzNsZuEO_sUvyt27T-w8xFbzd", output="datasets/deepglobe.zip", quiet=False)
  ```

---

## 3. Cách dùng nhanh trong Notebook / Python:
```python
DATASET_DRIVE_IDS = {
    "deepglobe": "1ktPxbmkNzNsZuEO_sUvyt27T-w8xFbzd",
    "fss1000": "1tt3dkdASjXt58t-2A9zeucZ397ZRF7In"
}
```
