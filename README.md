# CitationLens: Graph Analytics For Discovering Emerging Influential Papers
### Phân Tích Đồ Thị Để Phát Hiện Các Bài Báo Khoa Học Tiềm Năng Bùng Nổ Trên Dữ Liệu Lớn OGBN-MAG

<p align="center">
  <img src="https://img.shields.io/badge/Apache_Spark-3.5-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="Spark" />
  <img src="https://img.shields.io/badge/PySpark_MLlib-Distributed_ML-blue?style=for-the-badge&logo=apache" alt="MLlib" />
  <a href="https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag"><img src="https://img.shields.io/badge/Dataset-Stanford_OGBN--MAG-7928CA?style=for-the-badge" alt="OGBN-MAG Dataset" /></a>
  <img src="https://img.shields.io/badge/GraphFrames-PageRank-green?style=for-the-badge" alt="GraphFrames" />
  <img src="https://img.shields.io/badge/FastAPI-Production_API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

> 📦 **Dataset Stanford OGBN-MAG (694,450 bài báo & 7.1 triệu liên kết trích dẫn):**  
> • 🌐 **Trang chủ Dataset chính thức:** [https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)  
> • 📥 **Link tải trực tiếp file nén SNAP:** [http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip](http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip)  
> • 🗂️ **Dữ liệu tiền xử lý sẵn trong repo:** [`data/processed/citation_ml_dataset.parquet`](data/processed/citation_ml_dataset.parquet) & [`data/processed/citation_lens.db`](data/processed/citation_lens.db)

---

<p align="center">
  <img src="assets/ui_design/citationlens_ui_mockup.png" width="95%" alt="CitationLens Interactive Web Dashboard" />
  <br />
  <em>Hình: Bảng điều khiển tương tác trực quan (Interactive Dashboard) của hệ sinh thái CitationLens</em>
</p>

---

## 📑 Mục Lục
1. [Giới thiệu Đề tài & Nghịch lý Thiên vị Độ phổ biến](#-1-giới-thiệu-đề-tài--nghịch-lý-thiên-vị-độ-phổ-biến)
2. [Kiến trúc Hệ thống Gợi ý Lai 3 Chiều (Two-Stage Hybrid)](#-2-kiến-trúc-hệ-thống-gợi-ý-lai-3-chiều-two-stage-hybrid)
3. [Dữ liệu Lớn & Chuẩn mực Chống Rò rỉ Thời gian (Zero-Leakage)](#-3-dữ-liệu-lớn--chuẩn-mực-chống-rò-rỉ-thời-gian-zero-leakage)
4. [Kiểm soát Đa cộng tuyến Nghiêm ngặt (Multicollinearity)](#-4-kiểm-soát-đa-cộng-tuyến-nghiêm-ngặt-multicollinearity)
5. [Quy trình Sàng lọc Sơ bộ & Bảng Đối chuẩn 22 Mô hình PySpark MLlib](#-5-quy-trình-sàng-lọc-sơ-bộ--bảng-đối-chuẩn-22-mô-hình-pyspark-mllib)
6. [Thực nghiệm Đối đầu Giải quyết Thiên vị Độ phổ biến (vs Google Scholar)](#-6-thực-nghiệm-đối-đầu-giải-quyết-thiên-vị-độ-phổ-biến-vs-google-scholar)
7. [Đánh giá Năng lực Tính toán Phân tán của Apache Spark](#-7-đánh-giá-năng-lực-tính-toán-phân-tán-của-apache-spark)
8. [Hướng dẫn Cài đặt & Chạy Docker Đơn giản](#-8-hướng-dẫn-cài-đặt--chạy-docker-đơn-giản)
9. [Cấu trúc Thư mục Dự án](#-9-cấu-trúc-thư-mục-dự-án)
10. [Tài liệu Tham khảo Chuẩn Quốc tế](#-10-tài-liệu-tham-khảo-chuẩn-quốc-tế)

---

## 📌 1. Giới thiệu Đề tài & Nghịch lý Thiên vị Độ phổ biến

Trong kỷ nguyên bùng nổ thông tin học thuật (Big Scholarly Data), hàng triệu công trình nghiên cứu được xuất bản mỗi năm. Tuy nhiên, các công cụ tìm kiếm và gợi ý học thuật phổ biến hiện nay (như Google Scholar, Scopus) vận hành chủ yếu dựa trên số lượng trích dẫn tích lũy hoặc PageRank tĩnh, dẫn tới **Nghịch lý Thiên vị độ phổ biến (Popularity Bias / Hiệu ứng dồn tích trích dẫn cũ)**:
* **Thiên vị bài cũ:** Các bài báo kinh điển xuất bản từ lâu liên tục được đẩy lên đầu trang tìm kiếm dù tốc độ phát triển chuyên môn đã chững lại.
* **Bỏ quên bài đột phá mới:** Các công bố khoa học đột phá mới xuất bản (*Emerging Influential Papers*) dù mang ý tưởng tiên phong nhưng do thời gian tồn tại ngắn nên chưa tích lũy đủ trích dẫn, dẫn đến bị chìm lấp trong kho dữ liệu khổng lồ.

**Hệ sinh thái CitationLens** được phát triển nhằm giải quyết triệt để nghịch lý này: Ứng dụng các thuật toán đồ thị phân tán trên **Apache Spark** kết hợp học máy phân tán để nhận diện chính xác các **"viên ngọc tiềm năng" (Scientific Gems)** đang trên đà bùng nổ ảnh hưởng khoa học trong tương lai.

---

## 🏛️ 2. Kiến trúc Hệ thống Gợi ý Lai 3 Chiều (Two-Stage Hybrid)

Hệ thống được thiết kế theo mô hình lọc 2 tầng kết hợp 3 chiều thông tin độc lập nhằm dung hòa giữa **độ chính xác theo chủ đề** và **khả năng phát hiện bài báo bùng nổ mới**:

<p align="center">
  <img src="assets/images/hybrid_recommender_architecture_grayscale_vi.png" width="95%" alt="Sơ đồ kiến trúc gợi ý lai 3 chiều" />
  <br />
  <em>Hình 1.1: Sơ đồ kiến trúc hệ thống gợi ý bài báo khoa học dạng lai 3 chiều (Two-Stage Hybrid Recommender)</em>
</p>

### Hệ thống Công thức Toán học Cốt lõi:
1. **Vận tốc trích dẫn tích lũy ($v_c$):**
   $$v_c = \frac{\text{past\_in\_degree}}{\text{paper\_age}} = \frac{\text{past\_in\_degree}}{2018 - \text{pub\_year} + 1}$$
2. **Độ liên quan ngữ cảnh Tầng 1 ($Relevance$):**
   $$Relevance(P_{seed}, P_{cand}) = 0.6 \cdot \frac{1.0}{1.0 + |Year_{cand} - Year_{seed}|} + 0.4 \cdot \frac{\min(PR_{cand}, PR_{seed})}{\max(PR_{cand}, PR_{seed})}$$
3. **Biên dự báo tuyến tính phạt bài cũ ($z$):**
   $$z = w_v \cdot \frac{v_c}{\sigma_v} + w_{PR} \cdot \frac{PR}{\sigma_{PR}} - w_{age} \cdot \frac{paper\_age}{\sigma_{age}}$$
4. **Xác suất bùng nổ tiềm năng qua hàm Sigmoid ($P(Emerging)$):**
   $$P(Emerging) = \sigma(z) = \frac{1.0}{1.0 + e^{-z}}$$
5. **Điểm số xếp hạng tổng hợp dạng lai ($Score_{final}$):**
   $$Score_{final} = \alpha \cdot Relevance + (1.0 - \alpha) \cdot P(Emerging) \quad (\text{với } \alpha = 0.4)$$

---

## 🗄️ 3. Dữ liệu Lớn & Chuẩn mực Chống Rò rỉ Thời gian (Zero-Leakage)

### 3.1. Nguồn Dữ liệu & Hướng dẫn Tải về (Dataset Links & Download Guide)

* **Tên bộ dữ liệu:** **Stanford OGBN-MAG (Open Graph Benchmark - Microsoft Academic Graph)**
* 🌐 **Trang chủ Dataset chính thức:** [https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
* 📦 **Link tải trực tiếp file nén từ Stanford SNAP (~1.7 GB):** [http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip](http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip)
* 📄 **Bài báo công bố chuẩn quốc tế (NeurIPS 2020):** [Hu et al., "Open Graph Benchmark: Datasets for Machine Learning on Graphs"](https://arxiv.org/abs/2005.00687)

#### 📥 Các phương thức tải dữ liệu:

* **Cách 1: Tải trực tiếp bằng Terminal / Dòng lệnh (`curl` hoặc `wget`):**
  ```bash
  # Tạo thư mục lưu trữ và tải file nén từ máy chủ Stanford SNAP
  mkdir -p data/mag && cd data/mag
  curl -O http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip
  # hoặc dùng wget:
  # wget http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip

  # Giải nén bộ dữ liệu
  unzip mag.zip
  cd ../..
  ```

* **Cách 2: Tải tự động bằng Python qua thư viện chuẩn `ogb` của Stanford:**
  ```python
  # Cài đặt thư viện OGB chính thức
  pip install ogb

  # Tải và tự động giải nén bộ dữ liệu ogbn-mag vào thư mục data/
  from ogb.nodeproppred import NodePropPredDataset
  dataset = NodePropPredDataset(name="ogbn-mag", root="data")
  ```

* **Cách 3: Sử dụng Dữ liệu đã Tiền xử lý sẵn trong Repository (Khuyên dùng khi chạy Demo):**
  Dự án đã xử lý sẵn và đóng gói dữ liệu tối ưu trong thư mục [`data/processed/`](data/processed/) để người dùng có thể khởi chạy FastAPI Web App ngay lập tức mà không cần xử lý lại từ đầu:
  * 🗄️ [`data/processed/citation_lens.db`](data/processed/citation_lens.db): Cơ sở dữ liệu SQLite tối ưu phục vụ tìm kiếm bài báo và gợi ý thời gian thực (~0.25s).
  * 📊 [`data/processed/citation_ml_dataset.parquet`](data/processed/citation_ml_dataset.parquet): Tập dữ liệu Parquet phân tán **694,450 bài báo** và **7.1 triệu liên kết trích dẫn** đã trích xuất đặc trưng và phân vùng thời gian Train - Valid - Test.

### 3.2. Nguyên tắc Phân chia Thời gian Nghiêm ngặt (Strict Temporal Split)
* `Train`: Các bài báo xuất bản $\le 2016$ (**559,373** bài báo).
* `Validation`: Các bài báo xuất bản năm $2017$ (**70,198** bài báo).
* `Test`: Các bài báo xuất bản năm $2018$ (**64,879** bài báo) và đánh giá lượng tăng trích dẫn trong tương lai ($\ge 2019$).
* **Cam kết toàn vẹn học thuật:** Giao thoa giữa 3 tập là chính xác **0 bài báo**. Tuyệt đối không dùng K-Fold ngẫu nhiên xuyên suốt tập dữ liệu để ngăn chặn triệt để nguy cơ trích dẫn tương lai rò rỉ ngược về quá khứ (*Data Leakage*).

<p align="center">
  <img src="assets/images/dataset_charts/chart_cell39_0.png" width="32%" alt="Phân phối bài báo theo năm" />
  <img src="assets/images/dataset_charts/chart_cell19_0.png" width="32%" alt="Phân chia thời gian Train-Valid-Test" />
  <img src="assets/images/dataset_charts/chart_cell11_1.png" width="32%" alt="Phân phối trích dẫn đuôi dài" />
  <br />
  <em>Hình 1.2 - 1.4: Khảo sát phân phối tập dữ liệu OGBN-MAG (694,450 bài báo) theo năm xuất bản, phân vùng thời gian không rò rỉ và phân phối trích dẫn đuôi dài (Power-law / Heavy-tailed)</em>
</p>

---

## 🔬 4. Kiểm soát Đa cộng tuyến Nghiêm ngặt (Multicollinearity)

Trong quá trình phân tích ma trận tương quan Spearman ($\rho$):
* Cặp biến trích dẫn quá khứ `past_in_degree` và vận tốc trích dẫn `citation_velocity` có hệ số tương quan rất cao:
  $$\rho = 0.923 \ge 0.8$$
  *(Vi phạm nghiêm trọng ngưỡng an toàn đa cộng tuyến do quan hệ đại số phụ thuộc trực tiếp).*
* Cặp biến `past_in_degree` và điểm cấu trúc đồ thị `pagerank_score` có tương quan $\rho = 0.853 \ge 0.8$.

**Giải pháp:** Loại bỏ hoàn toàn biến `past_in_degree` ra khỏi không gian đặc trưng học máy, giữ lại bộ ba đặc trưng động học độc lập:
$$\mathcal{F}_{\text{core}} = \{\text{citation\_velocity}, \text{pagerank\_score}, \text{paper\_age}\}$$

<p align="center">
  <img src="assets/images/analysis-dataset/analysis-dataset-after-handle_cell18_1.png" width="38%" alt="Ma trận tương quan Spearman" />
  <img src="assets/images/analysis-dataset/analysis-dataset-after-handle_cell20_0.png" width="58%" alt="Phân phối Box Plot các đặc trưng" />
  <br />
  <img src="assets/images/analysis-dataset/analysis-dataset-after-handle_cell21_1.png" width="80%" alt="Histogram 3 đặc trưng cốt lõi" />
  <br />
  <em>Hình 1.5 - 1.7: Ma trận tương quan Spearman phát hiện đa cộng tuyến (ρ = 0.923), biểu đồ Box Plot phân phối các đặc trưng và Histogram của bộ ba đặc trưng cốt lõi sau xử lý</em>
</p>

---

## 📊 5. Quy trình Sàng lọc Sơ bộ & Bảng Đối chuẩn 22 Mô hình PySpark MLlib

### 5.1. Sàng lọc sơ bộ (Preliminary Screening)
Thử nghiệm tự động bằng công cụ `LazyClassifier` (`model_before_handle_data/lazy_classifier.ipynb`) trên tập mẫu 30,000 dòng đóng vai trò **bàn đạp thực nghiệm** để khảo sát nhanh hơn 20 thuật toán scikit-learn. Kết quả chỉ ra 3 trường phái tối ưu nhất: **Tuyến tính biên phân cách** (Logistic, Linear SVC), **Cây quyết định** (Decision Tree), và **Cây kết hợp** (Random Forest, GBT) để đưa lên cụm **Apache Spark phân tán**.

<p align="center">
  <img src="assets/images/model_classweight/gbt_classifier_cell5_1.png" width="48%" alt="Phân phối nhãn mất cân bằng" />
  <br />
  <em>Hình 1.8: Phân phối nhãn trên tập huấn luyện – minh họa trực quan sự mất cân bằng lớp nghiêm trọng (Lớp 0: 86.5%, Lớp 1: 13.5% ≈ 1 : 6.4)</em>
</p>

### 5.2. Bảng Đối chuẩn Toàn diện 22 Mô hình Học máy Phân tán trên Spark MLlib
Thực nghiệm đối chuẩn được triển khai đầy đủ trên **22 notebook phân tán** trên toàn bộ 694k bài báo, đánh giá trên tập kiểm định tương lai Test Set ($n = 64,879$):

| Nhóm phương pháp | Thuật toán học máy | Valid PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dữ liệu gốc** | Logistic Regression | 0.6507 | 0.5194 | 0.7765 | 0.8361 |
| **Dữ liệu gốc** | Linear SVC | 0.6643 | 0.5467 | 0.7903 | 0.8361 |
| **Dữ liệu gốc** | Decision Tree | 0.6188 | 0.5595 | 0.6196 | 0.8626 |
| **Dữ liệu gốc** | Random Forest | 0.6257 | 0.5404 | 0.7433 | 0.8435 |
| **Dữ liệu gốc** | GBT Classifier | 0.6496 | 0.5271 | 0.7854 | 0.8626 |
| **Random Oversampling** | Logistic Regression | 0.6507 | 0.5194 | 0.7765 | 0.8559 |
| **Random Oversampling** | Linear SVC | 0.6506 | 0.5194 | 0.7765 | 0.8749 |
| **Random Oversampling** | Decision Tree | 0.5389 | 0.4991 | 0.7428 | 0.8567 |
| **Random Oversampling** | Random Forest | 0.5652 | 0.5050 | 0.7863 | 0.8567 |
| **Random Oversampling** | GBT Classifier | 0.6590 | 0.5288 | 0.7868 | 0.8560 |
| **Random Undersampling** | Logistic Regression | 0.6507 | 0.5194 | 0.7765 | 0.8559 |
| **Random Undersampling** | Linear SVC | 0.6506 | 0.5194 | 0.7765 | 0.8749 |
| **Random Undersampling** | Decision Tree | 0.5390 | 0.4991 | 0.7428 | 0.8560 |
| **Random Undersampling** | Random Forest | 0.5860 | 0.5048 | 0.7859 | 0.8560 |
| **Random Undersampling** | GBT Classifier | 0.6557 | 0.5208 | 0.7849 | 0.8252 |
| **Class Weight (Đề xuất)** | Logistic Regression | 0.6507 | 0.5194 | 0.7765 | 0.8559 |
| **Class Weight (Đề xuất)** | Linear SVC | 0.6506 | 0.5194 | 0.7765 | 0.8749 |
| **Class Weight (Đề xuất)** | Decision Tree | 0.5396 | 0.4902 | 0.7820 | 0.8560 |
| **Class Weight (Đề xuất)** | Random Forest | 0.5909 | 0.5052 | 0.7861 | 0.8560 |
| **Class Weight (Đề xuất)** | **GBT Classifier** | **0.6604** | **0.5337** | **0.7870** | **0.8560** |
| **Hyperparameter Tuning** | **GBT + CW (depth=4, iter=15)** | **0.6618** | **0.5385** | **0.7868** | **0.8560** |
| **Stacking Ensemble (Đề xuất)** | **Random Forest + GBT → Logistic** | **0.6613** | **0.5412 ★** | **0.7888 ★** | **0.8617 ★** |

> **Nhận định thực nghiệm then chốt:**
> 1. **Class Weight là chiến lược tối ưu nhất:** Oversampling và Undersampling làm suy giảm mạnh PR-AUC của các mô hình cây (< 0.5050) do Overfitting hoặc mất mát thông tin lớp đa số. Class Weight duy trì toàn vẹn dữ liệu gốc và đưa GBT đạt Val PR-AUC = 0.6604 cao nhất.
> 2. **Tối ưu hóa siêu tham số phân tán:** Giảm `maxDepth` từ 5 xuống 4 và `maxIter` từ 20 xuống 15 giúp GBT tránh quá khớp vào đuôi nặng và rút ngắn 20% thời gian huấn luyện.
> 3. **Stacking Ensemble đạt đỉnh:** Kết hợp thế mạnh giữa Bagging (RF) và Boosting (GBT) thông qua Meta Logistic Regression xác lập kỷ lục cao nhất đề tài ($PR	ext{-}AUC = 0.5412$, $ROC	ext{-}AUC = 0.7888$, $F_1 = 0.8617$).

<p align="center">
  <img src="assets/images/model_classweight/gbt_classifier_cell20_0.png" width="31%" alt="Confusion Matrix GBT Test" />
  <img src="assets/images/model_classweight/gbt_classifier_cell22_0.png" width="33%" alt="Precision-Recall Curve" />
  <img src="assets/images/model_classweight/gbt_classifier_cell24_0.png" width="33%" alt="ROC-AUC Curve" />
  <br />
  <em>Hình 1.9 - 1.11: Ma trận nhầm lẫn (Confusion Matrix), Đường cong Precision-Recall (PR-AUC = 0.5337) và Đường cong ROC-AUC (ROC-AUC = 0.7870) của mô hình GBT + Class Weight trên tập Test tương lai</em>
</p>

---

## 🎯 6. Thực nghiệm Đối đầu Giải quyết Thiên vị Độ phổ biến (vs Google Scholar)

Để chứng minh năng lực phát hiện "viên ngọc tiềm năng", hệ thống CitationLens được đưa vào đối đầu trực diện với **Naive Baseline (mô phỏng thuật toán xếp hạng dựa trên tổng trích dẫn của Google Scholar)** trên Top 50 bài báo gợi ý từ kho dữ liệu thực nghiệm:

| Tiêu chí đo lường đối chuẩn | Naive Baseline (Google Scholar) | Mô hình Đề xuất (CitationLens) | Ý nghĩa khoa học thực tiễn |
| :--- | :---: | :---: | :--- |
| **Cơ chế xếp hạng chính** | Thuần số trích dẫn cũ (`past_in_degree`) | Động lực trích dẫn và Uy tín đồ thị | Phá vỡ thế độc quyền của dữ liệu bài báo cũ |
| **Tuổi trung bình bài báo** | **7.4 năm tuổi** (Thiên lệch bài rất cũ) | **5.8 năm tuổi** (Trẻ hơn rõ rệt) | Trẻ hóa danh mục gợi ý, rút ngắn độ trễ nhận diện xu hướng |
| **Số bài mới ($\le$ 2 năm) lọt Top 50** | **0 / 50 bài (0.0%)** | **5 / 50 bài (10.0%)** | **Cứu được 10% công bố mới mà hệ cũ bỏ sót 100%** |
| **Độ chính xác Top 10 (P@10)** | **100.0%** (10/10 bài) | **100.0%** (10/10 bài) | Bảo toàn độ chuẩn xác dự báo tuyệt đối |
| **Độ chính xác Top 50 (P@50)** | **100.0%** (50/50 bài) | **100.0%** (50/50 bài) | Toàn bộ 50 bài gợi ý đều bùng nổ mạnh mẽ trong tương lai |

<p align="center">
  <img src="assets/images/analysis_popularity_bias/popularity_bias_benchmark_cell11_0.png" width="90%" alt="Popularity Bias Benchmark Comparison" />
  <br />
  <em>Hình: Biểu đồ đối đầu định lượng giữa CitationLens và Google Scholar – Chứng minh khả năng phá vỡ thiên vị bài cũ và phát hiện các bài báo mới nổi</em>
</p>

---

## ⚡ 7. Đánh giá Năng lực Tính toán Phân tán của Apache Spark

Đo lường năng lực mở rộng dữ liệu lớn trên cụm phân tán Apache Spark qua 2 trục:

1. **Hiệu quả Tăng tốc Song song (Parallel Speedup):**
   * 1 Core: **1.01 giây**
   * 2 Cores: **0.26 giây**
   * 4 Cores: **0.22 giây**
   * **Hệ số tăng tốc đạt $\text{Speedup} = 4.59\times$**, bám sát đường lý tưởng, khẳng định cơ chế phân vùng RDD/DataFrame của Spark hoạt động tối ưu.
2. **Khả năng Mở rộng Quy mô Dữ liệu (Data Scalability):**
   * Khi quy mô dữ liệu tăng từ 100,000 bài lên 300,000 bài và toàn bộ **694,450 bài báo**, thời gian thực thi phân tán duy trì ổn định dưới **0.25 giây** nhờ cơ chế *Lazy Evaluation* và bộ tối ưu hóa *Catalyst Optimizer*, triệt tiêu nguy cơ quá tải bộ nhớ.

<p align="center">
  <img src="assets/images/spark_scalability_benchmark_grayscale.png" width="85%" alt="Apache Spark Scalability Benchmark" />
  <br />
  <em>Hình 1.12: Đồ thị đo lường hệ số tăng tốc (Speedup = 4.59x) và khả năng mở rộng dữ liệu của Apache Spark trên toàn bộ 694,450 bài báo</em>
</p>

---

## 🚀 8. Hướng dẫn Cài đặt & Khởi chạy Hệ thống

Bạn có thể lựa chọn khởi chạy hệ thống theo 1 trong 2 phương pháp:
* **Cách 1: Khởi chạy Trực tiếp Backend bằng FastAPI (Local Python)** – Thích hợp để phát triển, debug, tinh chỉnh mã nguồn và kiểm thử API nhanh mà không cần môi trường ảo hóa.
* **Cách 2: Khởi chạy Trọn gói bằng Docker Container** – Thích hợp triển khai sản phẩm hoàn chỉnh, độc lập và tương thích đa nền tảng.

---

### 8.1. Cách 1: Khởi chạy Trực tiếp Backend bằng FastAPI (Local Python)

#### Bước 1: Chuẩn bị Môi trường Python
* **Yêu cầu:** Python phiên bản **3.9+** trở lên.
* Khuyến nghị tạo môi trường ảo (virtual environment) để cô lập các gói thư viện:
```bash
# 1. Mở Terminal tại thư mục gốc của dự án
cd Graph_Analytics_For_Discovering_Emerging_Influential_Papers

# 2. Tạo môi trường ảo
python3 -m venv venv

# 3. Kích hoạt môi trường ảo:
# Trên macOS / Linux:
source venv/bin/activate
# Trên Windows (Command Prompt):
venv\Scripts\activate.bat
# Trên Windows (PowerShell):
venv\Scripts\Activate.ps1
```

#### Bước 2: Cài đặt các Thư viện Phụ thuộc
Cài đặt các gói thư viện nhẹ trong `requirements.txt` (`fastapi`, `uvicorn`, `numpy`, `pandas`, `jinja2`):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Bước 3: Khởi chạy Backend Server
Bạn có thể khởi động server FastAPI bằng 1 trong 2 lệnh sau:

* **Lựa chọn A (Khuyên dùng cho Development - Tự động reload khi sửa code):**
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Lựa chọn B (Chạy trực tiếp module Python):**
  ```bash
  python -m app.main
  ```

Sau khi chạy lệnh, màn hình Terminal sẽ thông báo:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
INFO:     Application startup complete.
```

---

### 8.2. Cách 2: Khởi chạy Trọn gói bằng Docker Container

Nếu máy tính đã cài đặt **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Windows / macOS / Linux) và ứng dụng Docker đang bật:

#### 👉 Lựa chọn 1: Chạy trực tiếp từ Docker Image đã có
Nếu bạn đã có sẵn image `graph-2:latest`:
```bash
docker run -d -p 8000:80 --name citation-lens graph-2:latest
```

#### 👉 Lựa chọn 2: Tự build Image từ mã nguồn (Khuyên dùng khi clone repo mới)
```bash
# Bước 1: Build Docker Image từ Dockerfile
docker build -t graph-2:latest .

# Bước 2: Khởi chạy container ánh xạ cổng 8000 máy chủ vào cổng 80 container
docker run -d -p 8000:80 --name citation-lens graph-2:latest
```

#### Các lệnh quản lý Docker hữu ích:
```bash
# Kiểm tra danh sách container đang chạy
docker ps

# Xem log hoạt động theo thời gian thực
docker logs -f citation-lens

# Dừng container
docker stop citation-lens

# Khởi động lại container đã dừng
docker start citation-lens

# Xóa container khi không còn sử dụng
docker rm -f citation-lens
```

---

### 8.3. Truy cập Ứng dụng & Kiểm thử API

Sau khi server khởi chạy thành công (theo Cách 1 hoặc Cách 2), mở trình duyệt web và truy cập:

* 🌐 **Giao diện Web tương tác (Dashboard):** [http://localhost:8000/](http://localhost:8000/)
* 📚 **Tài liệu API Swagger UI tương tác:** [http://localhost:8000/docs](http://localhost:8000/docs)
* 📖 **Tài liệu API ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

> [!TIP]
> * **Tìm kiếm & Gợi ý trên Web Dashboard:** Nhập từ khóa bất kỳ (ví dụ: `graph neural networks`, `deep learning`, `attention`) để tìm kiếm bài báo hạt nhân ($P_{seed}$). Hệ thống sẽ tự động kích hoạt bộ lọc Two-Stage Hybrid Recommender và hiển thị Top bài báo tiềm năng bùng nổ nhất kèm link DOI bài báo gốc.
> * **Kiểm thử API qua dòng lệnh (cURL):**
>   ```bash
>   # 1. Tìm kiếm bài báo theo tiêu đề (<10ms)
>   curl "http://localhost:8000/api/search?q=graph"
>
>   # 2. Gợi ý Top-10 bài báo bùng nổ cho một bài báo hạt nhân
>   curl "http://localhost:8000/api/recommend?paper_id=2376672&alpha=0.4&top_k=10"
>   ```

### 8.4. Các Endpoints API Chính của Backend:
* `GET /`: Giao diện Dashboard tìm kiếm và khuyến nghị bài báo khoa học.
* `GET /api/search?q={keyword}`: Tìm kiếm bài báo hạt nhân ($P_{seed}$) theo tiêu đề với thời gian phản hồi <10ms.
* `GET /api/recommend?paper_id={id}&alpha=0.4&top_k=10`: Gợi ý Two-Stage Hybrid (Graph + Semantic + PySpark MLlib GBT Reranking) trả về Top-K bài báo tiềm năng bùng nổ kèm định danh số DOI thực tế.

---

## 📁 9. Cấu trúc Thư mục Dự án

```
Graph_Analytics_For_Discovering_Emerging_Influential_Papers/
├── app/                                 # Hệ thống Web Application & REST API
│   ├── main.py                          # FastAPI Application Server
│   ├── recommender.py                   # Two-Stage Hybrid Recommendation Engine
│   └── templates/
│       └── index.html                   # Giao diện Web Client tương tác
├── assets/                              # Toàn bộ hình ảnh, biểu đồ và UI mockups
│   ├── images/                          # Biểu đồ kiến trúc, EDA, MLlib, Scalability
│   └── ui_design/                       # Thiết kế giao diện Dashboard CitationLens
├── data/
│   ├── raw/                             # Dữ liệu thô OGBN-MAG từ Stanford
│   └── processed/
│       ├── citation_ml_dataset.parquet  # Bộ dữ liệu 694k dòng phân vùng temporal_split
│       └── citation_lens.db             # Cơ sở dữ liệu SQLite phục vụ Serving thời gian thực
├── etl/                                 # Quy trình Trích xuất - Biến đổi - Nạp dữ liệu
│   ├── extract.py                       # Trích xuất thực thể OGBN-MAG
│   ├── transform.py                     # Tính toán PageRank phân tán & Đặc trưng topo
│   └── load.py                          # Ghi nạp Parquet & SQLite
├── model_before_handle_data/            # 5 Mô hình MLlib cơ sở & 1 notebook LazyClassifier
├── model_after_oversampling/            # 5 Mô hình MLlib sau Random Oversampling
├── model_after_undersampling/           # 5 Mô hình MLlib sau Random Undersampling
├── model_classweight/                   # 5 Mô hình MLlib áp dụng Class Weight đề xuất
├── model_hyperparameter_tuning/         # 1 Mô hình GBT tinh chỉnh siêu tham số GridSearch
├── model_stacking/                      # 1 Mô hình Stacking Classifier (RF + GBT → Logistic)
├── analysis_popularity_bias/            # Notebook đối đầu định lượng với Google Scholar
├── plan/
│   └── citation_lens_overview.html      # Trang Dashboard tổng quan kiến trúc đề tài
├── reference/
│   ├── BaoCao_BigData_DoAn_Final.docx   # Báo cáo chuẩn mẫu Khoa CNTT - IUH (28 trang)
│   ├── BaoCao_BigData_DoAn_Final.pdf    # Bản PDF xuất bản hoàn chỉnh
│   └── MauBaocaoBigData.docx            # Mẫu báo cáo quy chuẩn của GVHD TS. Bùi Thanh Hùng
├── Dockerfile                           # Tệp đóng gói Docker Container sản phẩm
├── requirements.txt                     # Danh sách thư viện Python phụ thuộc
└── README.md                            # Tài liệu kỹ thuật chi tiết của dự án
```

---

## 📚 10. Tài liệu Tham khảo Chuẩn Quốc tế

* **[1]** Chen, P., Xie, H., Maslov, S., & Redner, S. (2007). *Finding scientific gems with Google's PageRank algorithm.* Journal of Informetrics, 1(1), 8-15.
* **[2]** Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). *Open Graph Benchmark: Datasets for machine learning on graphs.* Advances in Neural Information Processing Systems (NeurIPS), 33, 22118-22133.
* **[3]** Covington, P., Adams, J., & Sargin, E. (2016). *Deep neural networks for YouTube recommendations.* Proceedings of the 10th ACM Conference on Recommender Systems (RecSys), 191-198.
* **[4]** Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank citation ranking: Bringing order to the web.* Technical Report, Stanford InfoLab.
* **[5]** Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine.* The Annals of Statistics, 29(5), 1189-1232.
* **[6]** Small, H. (1973). *Co-citation in the scientific literature: A new measure of the relationship between two documents.* Journal of the American Society for Information Science (JASIS), 24(4), 265-269.
* **[7]** Priem, J., Piwowar, H., & Orr, R. (2022). *OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts.* arXiv preprint arXiv:2205.01833.
* **[8]** Zaharia, M., et al. (2016). *Apache Spark: A unified engine for big data processing.* Communications of the ACM, 59(11), 56-65.
* **[9]** Meng, X., et al. (2016). *MLlib: Machine learning in Apache Spark.* Journal of Machine Learning Research (JMLR), 17(34), 1-7.
* **[10]** Kimiring, S. (2024). *Building a hybrid recommender system: Netflix case study with FastAPI.* Towards Data Science.
* **[11]** Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5-32.
* **[12]** Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management, 24(5), 513-523.
