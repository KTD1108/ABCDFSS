# TÀI LIỆU TOÀN DIỆN VỀ BÀI BÁO GỐC (CVPR 2024) VÀ CẢI TIẾN ĐỀ XUẤT (E0–E3)
## Đề Tài: Cross-Domain Few-Shot Semantic Segmentation (CD-FSS)
### Dành cho người mới bắt đầu: Từ khái niệm trực quan đến toàn bộ công thức toán học

---

## MỤC LỤC
1. [Khái Niệm Cốt Lõi Cho Người Mới Bắt Đầu](#1-khái-niệm-cốt-lõi-cho-người-mới-bắt-đầu)
2. [Bài Báo Gốc Làm Gì? (Kiến Trúc & Phương Pháp Gốc CVPR 2024)](#2-bài-báo-gốc-làm-gì-kiến-trúc--phương-pháp-gốc-cvpr-2024)
3. [Hiện Tại Chúng Ta Thay Đổi Gì? (Hệ Thống 4 Cấu Hình E0 – E3)](#3-hiện-tại-chúng-ta-thay-đổi-gì-hệ-thống-4-cấu-hình-e0--e3)
4. [Toàn Bộ Công Thức Toán Học Chi Tiết](#4-toàn-bộ-công-thức-toán-học-chi-tiết)
   - [4.1. Định dạng bài toán Episode](#41-định-dạng-bài-toán-episode)
   - [4.2. Trích xuất đặc trưng đa tầng (Multi-layer Feature Extraction)](#42-trích-xuất-đặc-trưng-đa-tầng-multi-layer-feature-extraction)
   - [4.3. Kiến trúc Adapter: Pointwise (Gốc) vs. Depthwise Separable (Cải tiến)](#43-kiến-trúc-adapter-pointwise-gốc-vs-depthwise-separable-cải-tiến)
   - [4.4. Hàm mất mát thích ứng lúc kiểm thử (Test-Time Adaptation Loss)](#44-hàm-mất-mát-thích-ứng-lúc-kiểm-thử-test-time-adaptation-loss)
   - [4.5. Ma trận ái lực dày đặc (Dense Cross-Attention)](#45-ma-trận-ái-lực-dày-đặc-dense-cross-attention)
   - [4.6. Cơ chế gộp tầng: Mean Fusion (Gốc) vs. Softmax Margin Fusion (Cải tiến)](#46-cơ-chế-gộp-tầng-mean-fusion-gốc-vs-softmax-margin-fusion-cải-tiến)
   - [4.7. Phân ngưỡng nhị phân thích ứng (Adaptive Thresholding)](#47-phân-ngưỡng-nhị-phân-thích-ứng-adaptive-thresholding)
   - [4.8. Các thước đo đánh giá: Cumulative mIoU vs. Mean Episode-IoU](#48-các-thước-đo-đánh-giá-cumulative-miou-vs-mean-episode-iou)
5. [So Sánh Thực Nghiệm & Phân Tích Ý Nghĩa Khoa Học](#5-so-sánh-thực-nghiệm--phân-tích-ý-nghĩa-khoa-học)
6. [Tóm Tắt Bằng Một Bức Tranh Toàn Cảnh](#6-tóm-tắt-bằng-một-bức-tranh-toàn-cảnh)

---

## 1. KHÁI NIỆM CỐT LÕI CHO NGƯỜI MỚI BẮT ĐẦU

### 1.1. Phân đoạn ngữ nghĩa (Semantic Segmentation) là gì?
- Thay vì chỉ nhận diện "trong ảnh có con chó" (Phân loại ảnh - Classification), hay "vẽ khung bao quanh con chó" (Phát hiện vật thể - Object Detection), **Semantic Segmentation** yêu cầu máy tính tô màu **từng pixel** thuộc về đối tượng.
- Pixel nào thuộc về vật thể mục tiêu (ví dụ: khối u phổi, mảng da bệnh, tòa nhà) thì được gán nhãn $1$ (Foreground), pixel nào là nền thì gán nhãn $0$ (Background).

### 1.2. Few-Shot Semantic Segmentation (FSS) là gì?
- Trong học sâu truyền thống, để phân đoạn được "khối u", bạn cần hàng nghìn tấm ảnh đã được bác sĩ khoanh vùng sẵn để huấn luyện mô hình.
- Trong **Few-Shot Learning**, bạn chỉ có **duy nhất 1 tấm ảnh mẫu (1-shot)** hoặc **vài tấm (K-shot)** có sẵn nhãn (gọi là tập hỗ trợ - **Support Set**). Nhiệm vụ của mô hình là nhìn vào tấm ảnh mẫu đó, hiểu được vật thể cần tìm trông như thế nào, và ngay lập tức phân đoạn vật thể tương tự trên một tấm ảnh mới toanh (gọi là tập truy vấn - **Query Set**).

### 1.3. Khó khăn tột cùng: Cross-Domain (CD-FSS)
- Trong FSS thông thường (In-domain): Mô hình được huấn luyện trên ảnh chụp đồ vật đời thường (xe hơi, máy bay) và lúc test cũng phân đoạn đồ vật đời thường (xe đạp, con mèo). Phong cách ảnh, màu sắc, kết cấu tương đồng.
- Trong **Cross-Domain FSS**: Mô hình được huấn luyện trước trên tập dữ liệu đời thường (ImageNet / COCO), nhưng lúc test lại bị bắt buộc đi phân đoạn:
  1. **Ảnh y tế tổn thương da** (`ISIC`)
  2. **Ảnh X-quang lồng ngực / tràn dịch màng phổi** (`Lung / CXR`)
  3. **Ảnh viễn thám / vệ tinh chụp từ vũ trụ** (`DeepGlobe`)
  4. **Ảnh sinh vật / thợ lặn dưới đáy biển bị mờ đục** (`SUIM`)
  5. **Tập 1000 đối tượng đa hình thái** (`FSS-1000`)
- **Vấn đề cốt tử**: Sự chênh lệch miền biểu diễn (**Domain Shift**) quá lớn khiến các đặc trưng mà mạng nơ-ron học từ ảnh đời thường bị sai lệch nghiêm trọng. Nếu cập nhật lại toàn bộ mạng với chỉ 1 tấm ảnh mẫu duy nhất thì mạng sẽ bị **quá khớp cực nặng (Severe Overfitting)**.

---

## 2. BÀI BÁO GỐC LÀM GÌ? (KIẾN TRÚC & PHƯƠNG PHÁP GỐC CVPR 2024)

Bài báo gốc (tên đề tài: *A Simple Baseline for Cross-Domain Few-Shot Semantic Segmentation* - CVPR 2024) đưa ra một chiến lược thông minh để giải quyết bài toán trên:

1. **Đóng băng hoàn toàn Backbone (Frozen ResNet-50)**:
   - Giữ nguyên toàn bộ trọng số được trích xuất từ 16 khối Bottleneck của mạng ResNet-50 đã huấn luyện trước. Không cập nhật bất kỳ trọng số nào của Backbone để tránh làm hỏng các biểu diễn tổng quát cơ bản.
   - Trích xuất đặc trưng **Pre-ReLU** (giá trị trước khi qua hàm kích hoạt ReLU) để giữ lại trọn vẹn dải giá trị âm/dương mang thông tin biên sắc nét.

2. **Gắn các bộ điều hợp siêu nhẹ (Lightweight Pointwise Adapters - Conv $1\times 1$)**:
   - Ở mỗi tầng trong số 13 tầng đặc trưng trên cùng (từ tầng $l_0=3$ đến tầng $15$), tác giả gắn một bộ điều hợp tí hon chỉ gồm tích chập $1\times 1$.
   - Mục đích: Nén số kênh lớn (256, 512, 1024, 2048) xuống còn đúng $64$ kênh, vừa đủ để giữ thông tin ngữ nghĩa mà không làm bùng nổ số lượng tham số.

3. **Tối ưu hóa trực tiếp ngay lúc kiểm thử (Test-Time Online Adaptation - Thuật toán 2)**:
   - Với mỗi bài toán (episode) gồm 1 ảnh Support và 1 ảnh Query, mô hình sinh ra các góc nhìn biến dạng (Augmented Views) bằng phép biến đổi Affine (xoay, kéo nghiêng/shear, làm mờ).
   - Mô hình chạy thuật toán Gradient Descent (SGD) trong **25 epochs** chỉ để cập nhật trọng số của các bộ Adapter $1\times 1$ sao cho:
     - Các pixel cùng vị trí tương ứng của ảnh gốc và ảnh biến dạng phải hút nhau (Học tương phản **Dense InfoNCE**).
     - Phân phối giá trị trung bình và phương sai không bị trôi dạt (**Keep Variance Loss**).
     - Prototype vùng mục tiêu (Foreground) phải tách xa Prototype vùng nền (Background) theo nhãn của Support (**Contrastive Prototype Loss**).

4. **Tính ma trận tương đồng điểm ảnh (Dense Cross-Attention)**:
   - So sánh từng pixel của ảnh Query với từng pixel của ảnh Support thông qua phép nhân tích vô hướng đã chuẩn hóa (Scaled Dot-Product).
   - Lấy ma trận ái lực này nhân với mặt nạ Ground Truth của Support để tạo ra bản đồ xác suất thô (Coarse Prediction Map) cho Query.

5. **Gộp trung bình đa tầng (Mean Layer Fusion)**:
   - Vì mỗi tầng của ResNet-50 trích xuất ở một độ phân giải khác nhau (tầng thấp bắt biên, tầng cao bắt ngữ nghĩa), tác giả nội suy tất cả về kích thước $50 \times 50$, sau đó **cộng trung bình phẳng (Uniform Average)** lại với nhau.

6. **Phân ngưỡng nhị phân lai (Hybrid Otsu-Mean Thresholding)**:
   - Tìm một ngưỡng cắt $\theta = \max(\text{Otsu}, \text{Mean})$ để biến bản đồ xác suất liên tục thành mặt nạ nhị phân $\{0, 1\}$.

---

## 3. HIỆN TẠI CHÚNG TA THAY ĐỔI GÌ? (HỆ THỐNG 4 CẤU HÌNH E0 – E3)

Để kiểm chứng xem kiến trúc của tác giả đã tối ưu chưa, hay việc cải tiến các thành phần cụ thể có thể nâng cao độ chính xác trên các miền dữ liệu khác nhau, dự án thiết lập ma trận **Ablation Study** gồm 4 cấu hình:

| Ký hiệu | Tên cấu hình | Adapter được sử dụng | Cơ chế gộp tầng (Fusion) | Mục đích nghiên cứu |
| :---: | :--- | :--- | :--- | :--- |
| **E0** | **Baseline Gốc** | Pointwise Conv $1\times 1$ | Mean Fusion (Cộng trung bình) | Tái lập chính xác 100% thuật toán của bài báo CVPR 2024 làm mốc chuẩn. |
| **E1** | **Adapter Ablation** | **Depthwise Separable Conv $3\times 3$** | Mean Fusion (Cộng trung bình) | Kiểm tra xem việc thêm ngữ cảnh không gian cục bộ ($3\times 3$) vào Adapter có giúp bắt biên tốt hơn không. |
| **E2** | **Fusion Ablation** | Pointwise Conv $1\times 1$ | **Softmax Margin Fusion** | Kiểm tra xem việc cho các tầng có biên tách biệt Foreground/Background lớn nhận trọng số cao hơn có tốt hơn cộng đều không. |
| **E3** | **Phương Pháp Đầy Đủ** | **Depthwise Separable Conv $3\times 3$** | **Softmax Margin Fusion** | Kết hợp đồng thời cả Adapter $3\times 3$ và Softmax Margin Fusion để khảo sát hiệu ứng cộng hưởng. |

---

## 4. TOÀN BỘ CÔNG THỨC TOÁN HỌC CHI TIẾT

Dưới đây là từng bước toán học được triển khai chính xác trong mã nguồn (`src/`).

### 4.1. Định dạng bài toán Episode
Mỗi bài toán kiểm thử là một episode $\mathcal{E}$:
$$\mathcal{E} = \left( \mathcal{S}, \mathcal{Q} \right)$$
- **Tập hỗ trợ (Support Set)**: $\mathcal{S} = \left\{ (I_s^k, M_s^k) \right\}_{k=1}^K$ với $K=1$ (1-shot).
  - $I_s \in \mathbb{R}^{3 \times H \times W}$: Ảnh hỗ trợ kích thước $400 \times 400 \times 3$.
  - $M_s \in \{0, 1\}^{H \times W}$: Mặt nạ nhị phân ground-truth của ảnh hỗ trợ ($1$ là foreground, $0$ là background).
- **Tập truy vấn (Query Set)**: $\mathcal{Q} = (I_q, M_q)$.
  - $I_q \in \mathbb{R}^{3 \times H \times W}$: Ảnh cần dự đoán.
  - $M_q \in \{0, 1\}^{H \times W}$: Mặt nạ ground-truth dùng để đối chiếu tính điểm IoU.

---

### 4.2. Trích xuất đặc trưng đa tầng (Multi-layer Feature Extraction)
Mạng Backbone $\Phi$ (ResNet-50) trích xuất đặc trưng từ $L=16$ khối Bottleneck:
$$F_l = \Phi_l(I), \quad l \in \{0, 1, \dots, 15\}$$
- Tác giả lấy giá trị **Pre-ReLU** ngay trước khi đi qua hàm kích hoạt:
  $$F_l = x_{\text{identity}} + \text{Conv}_{\text{bottleneck}}(x_{\text{in}})$$
- Chỉ các tầng từ $l_0 = 3$ đến $l = 15$ ($13$ tầng) được sử dụng để suy luận (bỏ qua 3 tầng đầu vì độ phân giải quá lớn và chứa quá nhiều nhiễu mức thấp).

---

### 4.3. Kiến trúc Adapter: Pointwise (Gốc) vs. Depthwise Separable (Cải tiến)

#### A. Pointwise Adapter (E0 & E2 - Bài báo gốc)
Sử dụng tích chập $1\times 1$ tác động độc lập lên từng pixel theo chiều kênh:
$$Z_l = W_{\text{proj}} * \text{ReLU}\left( \text{BatchNorm}\left( W_{\text{in}} * F_l \right) \right)$$
- $W_{\text{in}} \in \mathbb{R}^{64 \times C_l \times 1 \times 1}$: Chiếu giảm số kênh từ $C_l \in \{256, 512, 1024, 2048\}$ xuống $64$.
- $W_{\text{proj}} \in \mathbb{R}^{64 \times 64 \times 1 \times 1}$: Chiếu tuyến tính không gian đặc trưng thích ứng.
- **Đặc điểm**: Cực kỳ ít tham số, tính toán hoàn toàn theo từng pixel riêng lẻ, không nhìn các pixel xung quanh.

#### B. Depthwise Separable Adapter $3\times 3$ với Residual Shortcut (E1 & E3 - Đề xuất)
Để nắm bắt được hình thái đường biên cục bộ mà không làm tăng quá nhiều tham số như phép tích chập $3\times 3$ thông thường ($C_{\text{in}} \times C_{\text{out}} \times 3 \times 3$), ta phân tách thành 2 bước:
1. **Depthwise Convolution $3\times 3$** (Tích chập theo từng kênh riêng rẽ, nhóm = $C_{\text{in}}$):
   $$\widetilde{F}_l = \text{ReLU}\left( \text{BatchNorm}\left( W_{\text{dw}} \star F_l \right) \right)$$
   với $W_{\text{dw}} \in \mathbb{R}^{C_l \times 1 \times 3 \times 3}$.
2. **Pointwise Convolution $1\times 1$** (Trộn thông tin giữa các kênh):
   $$\widehat{F}_l = \text{ReLU}\left( \text{BatchNorm}\left( W_{\text{pw}} * \widetilde{F}_l \right) \right)$$
   với $W_{\text{pw}} \in \mathbb{R}^{64 \times C_l \times 1 \times 1}$.
3. **Phép chiếu đầu ra & Đường tắt phần dư (Residual Shortcut)**:
   $$Z_l = W_{\text{out}} * \widehat{F}_l + W_{\text{res}} * F_l$$
   với $W_{\text{res}} \in \mathbb{R}^{64 \times C_l \times 1 \times 1}$ giúp bảo toàn dòng gradient, chống biến mất gradient khi tối ưu trong vài epoch ngắn.

---

### 4.4. Hàm mất mát thích ứng lúc kiểm thử (Test-Time Adaptation Loss)

Trong 25 epochs tối ưu hóa lúc test, với mỗi ảnh $I$, ta sinh ra ảnh biến đổi $I' = \mathcal{T}(I)$ qua phép biến đổi Affine ngẫu nhiên (xoay, kéo dãn góc nghiêng $\pm 20^\circ$) và làm mờ Gaussian.

Feature gốc $F$ được căn chỉnh không gian qua toán tử $\mathcal{A}_{\mathcal{T}}$ sao cho toạ độ $(u, v)$ của $F$ trùng khớp không gian với $F' = \Phi(I')$.
Sau khi đi qua Adapter, ta chuẩn hoá L2 các vector đặc trưng:
$$z_i = \frac{Z_{l}(u, v)}{\|Z_l(u, v)\|_2}, \quad z'_i = \frac{Z'_{l}(u, v)}{\|Z'_l(u, v)\|_2}$$

#### 1. Mất mát tương phản điểm ảnh dày đặc (Dense InfoNCE Loss)
Với mỗi vị trí pixel $i \in \{1, \dots, H_l W_l\}$, cặp dương (positive pair) là $(z_i, z'_i)$ (cùng vị trí không gian), còn các pixel khác $j \neq i$ là cặp âm (negative pairs):
$$\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{H_l W_l} \sum_{i=1}^{H_l W_l} \log \frac{\exp\left( \frac{z_i \cdot z'_i}{\tau} \right)}{\sum_{j=1}^{H_l W_l} \exp\left( \frac{z_i \cdot z'_j}{\tau} \right)}$$
với nhiệt độ $\tau = 0.5$.

#### 2. Mất mát điều hòa phân phối đặc trưng (Keep Variance Loss)
Ngăn chặn hiện tượng sụp đổ biểu diễn (feature collapse), ép phân phối trung bình ($\mu$) và phương sai ($\sigma^2$) giữa biểu diễn gốc và biểu diễn biến đổi phải khớp nhau:
$$\mathcal{L}_{\text{var}} = \frac{1}{C} \sum_{c=1}^C \left| \mu(Z_c) - \mu(Z'_c) \right| + \frac{1}{C} \sum_{c=1}^C \left| \sigma^2(Z_c) - \sigma^2(Z'_c) \right|$$

#### 3. Mất mát căn chỉnh Prototype hỗ trợ (Contrastive Prototype Loss)
Dựa vào mặt nạ hỗ trợ $M_s$, ta tính vector đại diện (Prototype) của vùng đối tượng $p_{\text{fg}}$ và vùng nền $p_{\text{bg}}$:
$$p_{\text{fg}} = \frac{\sum_{i} z_{s, i} \cdot M_{s, i}}{\sum_i M_{s, i} + \epsilon}, \quad p_{\text{bg}} = \frac{\sum_i z_{s, i} \cdot (1 - M_{s, i})}{\sum_i (1 - M_{s, i}) + \epsilon}$$
Tương tự tính $p'_{\text{fg}}, p'_{\text{bg}}$ từ ảnh biến đổi. Hàm mất mát ép $p_{\text{fg}}$ phải tương đồng với $p'_{\text{fg}}$ và tách xa $p'_{\text{bg}}$:
$$\mathcal{L}_{\text{proto}} = - \log \frac{\exp\left( \cos(p_{\text{fg}}, p'_{\text{fg}}) \right)}{\exp\left( \cos(p_{\text{fg}}, p'_{\text{fg}}) \right) + \exp\left( \cos(p_{\text{fg}}, p'_{\text{bg}}) \right) + \epsilon}$$

#### Tổng hàm mất mát tối ưu hóa cho mỗi tầng:
$$\mathcal{L}_{\text{total}} = \left( \mathcal{L}_{\text{InfoNCE}}^q + \mathcal{L}_{\text{var}}^q \right) + \left( \mathcal{L}_{\text{InfoNCE}}^s + \mathcal{L}_{\text{var}}^s \right) + \mathcal{L}_{\text{proto}}$$
Mô hình cập nhật trọng số của Adapter qua SGD:
$$\Theta_{\text{adapter}} \leftarrow \Theta_{\text{adapter}} - \eta \nabla_{\Theta} \mathcal{L}_{\text{total}}, \quad (\text{với } \eta = 0.01, \text{epoch} = 25)$$

---

### 4.5. Ma trận ái lực dày đặc (Dense Cross-Attention)

Sau khi có đặc trưng đã thích ứng $Z_q^{(l)} \in \mathbb{R}^{64 \times H_q \times W_q}$ và $Z_s^{(l)} \in \mathbb{R}^{64 \times H_s \times W_s}$:
1. Trải phẳng không gian thành ma trận:
   - Query: $Q \in \mathbb{R}^{(H_q W_q) \times 64}$
   - Key: $K \in \mathbb{R}^{(H_s W_s) \times 64}$
2. Tính ma trận ái lực tương quan (Affinity Matrix) bằng tích vô hướng chia tỷ lệ:
   $$A = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{(H_q W_q) \times (H_s W_s)}, \quad (\text{với } d_k = 64)$$
3. Chiếu mặt nạ Ground Truth của Support $M_s$ (đã được nội suy song tuyến tính bilinear về kích thước $(H_s, W_s)$ và duỗi phẳng thành vector $V \in [0, 1]^{(H_s W_s) \times 1}$):
   $$\hat{q}^{(l)} = A \cdot V \in [0, 1]^{(H_q W_q) \times 1}$$
4. Định hình lại (reshape) thành bản đồ xác suất thô: $\hat{q}^{(l)} \in [0, 1]^{H_q \times W_q}$.

---

### 4.6. Cơ chế gộp tầng: Mean Fusion (Gốc) vs. Softmax Margin Fusion (Cải tiến)

Sau khi tính được $13$ bản đồ dự đoán $\hat{q}^{(l)}$ từ các tầng $l \in [l_0, 15]$, toàn bộ được nội suy song tuyến tính về kích thước cơ sở của tầng $l_0$ (tức $50 \times 50$).

#### A. Mean Fusion (E0 & E1 - Bài báo gốc)
Gộp trung bình đồng đều tất cả các tầng:
$$\hat{q}_{\text{fused}} = \frac{1}{16 - l_0} \sum_{l=l_0}^{15} \hat{q}^{(l)}$$
- *Hạn chế*: Coi trọng số của mọi tầng như nhau, kể cả những tầng đặc trưng bị nhiễu hoặc không phân biệt được đối tượng trên miền dữ liệu mới.

#### B. Softmax Margin Fusion (E2 & E3 - Đề xuất)
Đánh giá độ phân tách (Margin) giữa biểu diễn đối tượng và biểu diễn nền của Support ở từng tầng $l$:
$$\delta_l = 1 - \cos\left( p_{\text{fg}}^{(l)}, p_{\text{bg}}^{(l)} \right) = 1 - \frac{p_{\text{fg}}^{(l)} \cdot p_{\text{bg}}^{(l)}}{\|p_{\text{fg}}^{(l)}\|_2 \|p_{\text{bg}}^{(l)}\|_2}$$
- Nếu ở tầng $l$, vector đối tượng $p_{\text{fg}}$ và nền $p_{\text{bg}}$ chỉ ngược chiều nhau (tách biệt cực tốt) $\to \cos \approx -1 \implies \delta_l \approx 2$ (Margin tối đa).
- Nếu đối tượng và nền bị nhập nhằng $\to \cos \approx 1 \implies \delta_l \approx 0$ (Margin tối thiểu).

Từ vector margin $[\delta_{l_0}, \dots, \delta_{15}]$, tính trọng số mềm (Softmax Weight) với tham số nhiệt độ $T=1.0$:
$$w_l = \frac{\exp\left( \frac{\delta_l}{T} \right)}{\sum_{j=l_0}^{15} \exp\left( \frac{\delta_j}{T} \right)}$$
Bản đồ dự đoán tổng hợp cuối cùng:
$$\hat{q}_{\text{fused}} = \sum_{l=l_0}^{15} w_l \cdot \hat{q}^{(l)}$$

---

### 4.7. Phân ngưỡng nhị phân thích ứng (Adaptive Thresholding)

Bản đồ $\hat{q}_{\text{fused}}$ được phóng to (bilinear upsample) về kích thước ảnh gốc $400 \times 400$.
Để chuyển thành nhãn nhị phân $\{0, 1\}$, ngưỡng cắt $\theta$ được tính theo phương pháp lai của tác giả:
$$\theta = \max\left( \text{Otsu}_{0.05}(\hat{q}_{\text{fused}}), \text{mean}(\hat{q}_{\text{fused}}) \right)$$
- $\text{Otsu}_{0.05}$: Thuật toán phân ngưỡng Otsu tự động cực đại hóa phương sai liên lớp (between-class variance), sau khi đã loại bỏ $5\%$ giá trị xác suất thấp nhất để loại nhiễu nền.
- Mặt nạ nhị phân dự đoán cuối cùng:
  $$\hat{M}_q(u, v) = \begin{cases} 1 & \text{nếu } \hat{q}_{\text{fused}}(u, v) > \theta \\ 0 & \text{ngược lại} \end{cases}$$

---

### 4.8. Các thước đo đánh giá: Cumulative mIoU vs. Mean Episode-IoU

Trong bài toán phân đoạn ảnh Few-Shot, có **hai cách tính IoU** hoàn toàn khác nhau cần phân biệt rõ ràng:

#### 1. Cumulative mIoU (Class-Aggregated mIoU)
Tập hợp toàn bộ diện tích giao (Intersection) và diện tích hợp (Union) của tất cả các episode thuộc về từng lớp ngữ nghĩa $c$:
$$\text{IoU}_c = \frac{\sum_{i \in \mathcal{E}_c} |P_i \cap G_i|}{\sum_{i \in \mathcal{E}_c} |P_i \cup G_i|}$$
Sau đó lấy trung bình cộng qua tất cả các lớp ngữ nghĩa xuất hiện:
$$\text{Cumulative mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c$$
- *Đặc điểm*: Các episode có diện tích vật thể lớn sẽ đóng góp trọng số lớn hơn; rất ổn định trước các episode có vật thể siêu nhỏ.

#### 2. Mean Episode-IoU (Macro-average per Episode)
Tính IoU riêng lẻ cho từng episode $i$, rồi lấy trung bình cộng đại số:
$$\text{IoU}^{(i)} = \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$
$$\text{Mean Episode-IoU} = \frac{1}{N} \sum_{i=1}^N \text{IoU}^{(i)}$$
- *Đặc điểm*: Mỗi episode được đối xử bình đẳng tuyệt đối bất kể vật thể to hay nhỏ.

---

## 5. SO SÁNH THỰC NGHIỆM & PHÂN TÍCH Ý NGHĨA KHOA HỌC

Kết quả thực nghiệm trên **20 runs độc lập (2.000 episodes)** trên Modal Cloud GPU (Tesla T4) với seed=42 được tổng hợp dưới đây:

### 5.1. Bảng kết quả Cumulative mIoU (%)
| Bộ dữ liệu | E0 (Baseline) | E1 (Adapter 3x3) | E2 (Margin Fusion) | E3 (Cải tiến kết hợp) | Xu hướng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DeepGlobe** (Ảnh vệ tinh) | **44.74%** | 44.08% | 44.44% | 43.73% | E0 tốt nhất |
| **ISIC** (Bệnh học da) | **43.29%** | 41.80% | 42.68% | 41.72% | E0 tốt nhất |
| **Lung** (X-quang lồng ngực) | 81.32% | 81.90% | 81.16% | **82.21%** | **E3 vượt trội (+0.89 pp)** |
| **FSS-1000** (Đa dạng 1000 lớp) | 69.85% | 65.73% | **69.89%** | 66.49% | E2 tối ưu biên |
| **SUIM** (Dưới nước) | **37.01%** | 35.16% | 36.68% | 35.35% | E0 tốt nhất |
| **Trung bình toàn bộ** | **55.24%** | 53.73% | 54.97% | 53.90% | Baseline cực kỳ kiên cố |

---

### 5.2. Ba bài học khoa học cốt lõi rút ra

#### Bài học 1: Tại sao Adapter $3\times 3$ (E1) lại bị giảm điểm ở hầu hết các dataset, nhưng lại THẮNG ở Lung (+0.58 pp)?
- **Lý do giảm ở DeepGlobe, ISIC, SUIM**: Tích chập $3\times 3$ tăng số lượng tham số tự do. Khi chỉ có **1 tấm ảnh support** trong điều kiện Test-Time Adaptation 25 epochs, mô hình bị **quá khớp (overfitting)** vào các chi tiết cục bộ ngẫu nhiên (nhiễu màu, đốm nước, vân đất). Pointwise Conv $1\times 1$ của bài báo gốc tuy đơn giản nhưng hoạt động như một bộ điều hòa (regularizer) hoàn hảo.
- **Lý do thắng ở Lung**: Ảnh X-quang phổi có cấu trúc giải phẫu học (anatomy) cực kỳ cố định và rõ ràng (xương sườn, vòm hoành, nhu mô phổi). Nhân tích chập $3\times 3$ nắm bắt hoàn hảo mối liên hệ không gian giữa bờ sườn và bờ phổi, giúp phân đoạn chính xác hơn đáng kể.

#### Bài học 2: Softmax Margin Fusion (E2) hoạt động ra sao?
- Margin Fusion duy trì hiệu năng rất sát baseline (-0.27 pp) và tạo ra bước tăng nhẹ trên FSS-1000 (+0.04 pp).
- Điều này chứng minh rằng việc gán trọng số tự động dựa trên độ tách biệt Foreground/Background là một cơ chế lành mạnh, không làm sụp đổ mô hình ngay cả trên các miền dữ liệu hoàn toàn xa lạ.

#### Bài học 3: Tương tác siêu cộng tính (Super-additive Interaction) ở E3 trên Lung
- Khi chạy riêng lẻ trên Lung:
  - $\Delta_{\text{Adapter}} = +0.58\text{ pp}$
  - $\Delta_{\text{Fusion}} = -0.16\text{ pp}$
  - Tổng tuyến tính dự kiến: $+0.58 + (-0.16) = +0.42\text{ pp}$.
- Tuy nhiên khi kết hợp cả hai ở **E3**:
  $$\Delta_{\text{Combined}} = \mathbf{+0.89\text{ pp}} > +0.42\text{ pp}$$
- Hiện tượng này chứng minh: Khi Adapter $3\times 3$ trích xuất được các đặc trưng không gian sắc nét hơn, Softmax Margin Fusion mới phát huy tối đa sức mạnh để khuếch đại các tầng đặc trưng chất lượng cao đó.

---

## 6. TÓM TẮT BẰNG MỘT BỨC TRANH TOÀN CẢNH

```text
                  [Support Image (1-shot)]              [Query Image]
                             │                                │
                             ▼                                ▼
                 ┌────────────────────────────────────────────────┐
                 │          Frozen Backbone (ResNet-50)           │
                 │        Trích xuất Pre-ReLU (16 tầng)           │
                 └────────────────────────────────────────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
              [E0, E2: Pointwise 1x1]       [E1, E3: Depthwise-Sep 3x3]
              (Siêu nhẹ, chống overfit)     (Bắt ngữ cảnh biên sắc nét)
                             │                             │
                             └──────────────┬──────────────┘
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │      Test-Time Online Adaptation (SGD)       │
                    │   25 epochs: InfoNCE + KeepVar + ProtoLoss   │
                    └──────────────────────────────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │     Dense Cross-Attention (Query x Support)  │
                    │         Q x K^T / sqrt(d)  x  V_mask         │
                    └──────────────────────────────────────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                    [E0, E1: Mean Fusion]       [E2, E3: Margin Fusion]
                    (Cộng đều 13 tầng)          (Trọng số Softmax theo margin)
                             │                             │
                             └──────────────┬──────────────┘
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │    Hybrid Thresholding: max(Otsu, Mean)      │
                    └──────────────────────────────────────────────┘
                                            │
                                            ▼
                               [Mặt Nạ Phân Đoạn Cuối Cùng]
```

Tài liệu này cung cấp toàn bộ nền tảng lý thuyết, toán học và thực nghiệm của đề tài. Mọi công thức đều khớp 100% với các dòng mã trong thư mục `src/` và báo cáo khoa học trong `docs/FULL_BENCHMARK_REPORT.md`.
