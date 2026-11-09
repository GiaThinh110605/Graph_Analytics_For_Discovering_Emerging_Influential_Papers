# Graph Analytics For Discovering Emerging Influential Papers
### Phân Tích Đồ Thị Để Phát Hiện Các Bài Báo Tiềm Năng Bùng Nổ

> **Đồ án môn học:** Big Data  
> **Giảng viên hướng dẫn:** TS. Bùi Thanh Hùng  
> **Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Công nghiệp TP.HCM (IUH)  
> **Nền tảng công nghệ:** Apache Spark, GraphFrames, PySpark MLlib, SQLite, Python  

---

## 📌 1. Giới thiệu Đề tài & Mục tiêu

Trong mạng lưới học thuật toàn cầu, hàng triệu bài báo khoa học được công bố mỗi năm nhưng chỉ một tỷ lệ nhỏ (~5%) tạo ra đột phá và định hình xu hướng nghiên cứu trong tương lai. Các công cụ tìm kiếm truyền thống (Google Scholar, Scopus) thường bị **thiên vị bài báo cũ (Popularity Bias)** do xếp hạng thuần theo số lượng trích dẫn lũy kế hoặc PageRank tĩnh, khiến các bài báo mới công bố có tiềm năng lớn bị chìm lấp.

**Mục tiêu của dự án:**
1. Khai phá mạng đồ thị trích dẫn có hướng (DAG) quy mô lớn trên nền tảng tính toán phân tán **Apache Spark**.
2. Thiết kế và trích xuất các đặc trưng động học topo mạng (PageRank, Citation Velocity, Paper Age).
3. Thử nghiệm và đánh giá đối đầu **5 mô hình học máy phân tán PySpark** qua **4 chiến lược xử lý dữ liệu mất cân bằng cực đoan (1 : 6.4)**.
4. Đề xuất mô hình tối ưu nhất (**GBT Classifier + Class Weight + Hyperparameter Tuning**) phục vụ cho hệ thống phát hiện và khuyến nghị bài báo tiềm năng bùng nổ (**Top-K Emerging Discovery**).

---

## 🗄️ 2. Dữ liệu & Kiến trúc Pipeline ETL Phân tán

* **Tập dữ liệu:** **OGBN-MAG (Open Graph Benchmark - Microsoft Academic Graph)** gồm **694,450 bài báo** và hàng triệu cạnh trích dẫn.
* **Quy chuẩn phân chia thời gian (Strict Temporal Split - Chống rò rỉ dữ liệu):**
  * `train`: Bài báo công bố $\le 2016$ (**559,373** dòng).
  * `valid`: Bài báo công bố năm $2017$ (**70,198** dòng).
  * `test`: Bài báo công bố năm $2018$ và đánh giá mức tăng trưởng trích dẫn $\ge 2019$ (**64,879** dòng: 56,072 nhãn 0 và 8,807 nhãn 1).
  * *Cam kết:* Tuyệt đối không dùng K-Fold ngẫu nhiên để tránh việc trích dẫn tương lai bị rò rỉ ngược về quá khứ (*Data Leakage*).

### Luồng xử lý dữ liệu (`etl/`):
```
[Dữ liệu thô MAG] ──> extract.py ──> transform.py (Spark + GraphFrames) ──> load.py
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          Parquet (Partitioned):                             SQLite Database:
   data/processed/citation_ml_dataset.parquet        data/processed/citation_lens.db
   (694,450 rows × 13 columns)                        (papers: 694k, top_emerging: 44.6k)
```

1. **`etl/extract.py`:** Trích xuất các thực thể phân tán: Năm xuất bản, Cạnh trích dẫn, Tác giả, Lĩnh vực, Vector 128D, và Split OGB.
2. **`etl/transform.py`:**
   * Tách luồng cạnh theo mốc thời gian: Cạnh quá khứ ($\le 2018$) và Cạnh tương lai ($= 2019$).
   * Chạy thuật toán **PageRank phân tán** bằng **GraphFrames** (`resetProbability=0.15, maxIter=10`) trên đồ thị cạnh quá khứ.
   * Tính toán các đặc trưng động học: Vận tốc trích dẫn `citation_velocity = past_in_degree / paper_age`.
   * Gán nhãn mục tiêu `label = 1.0` nếu độ tăng trích dẫn tương lai $\Delta c \ge \theta_{p95}$ (thuộc Top 5% bùng nổ của tập train).
3. **`etl/load.py`:** Ghi dữ liệu phân vùng ra Parquet và nạp vào SQLite phục vụ ứng dụng.

---

## 🔬 3. Xử lý Đa cộng tuyến (Multicollinearity Elimination)

Trong `analysis-dataset/analysis-dataset-after-handle.ipynb`, phân tích tương quan hạng Spearman ($\rho$) trên toàn bộ các biến ứng viên:
* Cặp biến `past_in_degree` và `citation_velocity` có tương quan cực cao: **$\rho = 0.923 \ge 0.8$** (Đa cộng tuyến nghiêm trọng do quan hệ đại số phụ thuộc trực tiếp).
* Cặp biến `past_in_degree` và `pagerank_score` có tương quan: **$\rho = 0.853 \ge 0.8$**.

**Giải pháp:** Áp dụng thuật toán Greedy Feature Elimination, **loại bỏ hoàn toàn biến `past_in_degree`**, chỉ giữ lại bộ 3 đặc trưng cốt lõi độc lập:
$$\mathcal{F}_{\text{core}} = \{\text{citation\_velocity}, \text{pagerank\_score}, \text{paper\_age}\}$$

---

## 📊 4. Thực nghiệm Đối đầu 5 Mô hình qua 4 Chiến lược Lệch lớp

Triển khai thực nghiệm đầy đủ trên **20 Jupyter Notebooks** sử dụng PySpark MLlib trên tập Test (64,879 bài báo):

| Nhóm Thư mục | Chiến lược Xử lý Dữ liệu | Mô hình tốt nhất | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `model_before_handle_data/` | Dữ liệu gốc (Imbalanced 1:6.4) | GBT Classifier | 24.87% | **78.19%** | 37.74% | 0.5271 | 0.7854 |
| `model_after_undersampling/`| Random Undersampling (RUS 1:1) | GBT Classifier | **67.28%** | 37.12% | 47.82% | 0.5050 | 0.7876 |
| `model_after_oversampling/` | Random Oversampling (ROS 1:1)  | GBT Classifier | 56.61% | 45.58% | 50.50% | 0.5337 | 0.7870 |
| **`model_classweight/`** *(Đề xuất)* | **Class Weight ($w_0=0.53, w_1=9.64$)** | **GBT Classifier** | **56.61%** | **45.58%** | **50.50%** | **0.5337** | **0.7870** |

> **Nhận xét chuyên môn:** 
> * Mô hình gốc bị dính **Nghịch lý độ chính xác (Accuracy Paradox)**: Accuracy cao (88.86%) nhưng bỏ lọt tới 75.13% bài báo bùng nổ (Recall chỉ 24.87%).
> * Chiến lược **Class Weight** là giải pháp tối ưu nhất: Không làm thay đổi kích thước dữ liệu (không làm chậm Spark), nâng Recall lên **56.61%** (bắt trúng **4,986 bài báo tiềm năng**), và đạt F1-Score cao nhất **50.50%**.

---

## ⚡ 5. Tinh chỉnh Siêu tham số (Hyperparameter Tuning)

Tại thư mục `model_hyperparameter_tuning/gbt_classifier.ipynb`:
* Ứng dụng **`ParamGridBuilder`** dò tìm tự động trên tập Validation theo chuẩn thời gian:
  * `maxDepth` $\in [4, 6]$, `maxIter` $\in [15, 20]$.
* **Phát hiện quan trọng:** Cấu hình **`maxDepth=4, maxIter=15`** đạt kết quả vượt trội:
  * **Test PR-AUC tăng vọt lên: 0.5416** (Kỷ lục cao nhất toàn bộ đề tài).
  * Giảm độ sâu cây từ 5 xuống 4 giúp cây tổng quát hóa tốt hơn, tránh overfit vào đuôi dài của dữ liệu.
  * Giảm số lượng cây từ 20 xuống 15 giúp **thời gian huấn luyện nhanh hơn ~20%** (chỉ mất ~5 giây).

---

## 🌐 6. Khắc phục Sự thiên lệch Đặc trưng (Stochastic GBT)

Tại thư mục `model_after_improve_data/gbt_classifier.ipynb`:
* Bổ sung 2 đặc trưng topo phi tuyến từ các nghiên cứu quốc tế:
  * $\text{pagerank\_velocity} = \frac{\text{pagerank\_score}}{\text{paper\_age}}$ *(Chen et al. 2007, Gori & Pucci 2007)*
  * $\text{prestige\_ratio} = \frac{\text{pagerank\_score}}{\ln(1 + \text{past\_in\_degree} + 1.0)}$ *(Franceschet 2010, Sayyadi & Getoor 2009)*
* Cấu hình **Stochastic GBT (Friedman, 2002)** với `featureSubsetStrategy="sqrt"` và `subsamplingRate=0.8`:
  * Tỷ trọng độc quyền của `citation_velocity` giảm từ **96.5% xuống 83.67%**.
  * Đóng góp của các đặc trưng đồ thị và uy tín tăng vọt lên **14.66%** (`pagerank_velocity`: 8.83%, `prestige_ratio`: 3.25%).
  * Chỉ số **Test ROC-AUC đạt đỉnh mới: 0.7899**.

---

## 🎯 7. Đánh giá Thực tế: Nghiệp vụ Top-K Discovery

Trong bài toán thực tế (Google Scholar / Semantic Scholar), người dùng không đọc toàn bộ 65,000 bài mà chỉ quan tâm đến các bài báo tiềm năng nhất. Đánh giá xếp hạng **Precision@K**:

| Nghiệp vụ Discovery | Số bài chọn ra | Số bài bùng nổ thực tế | **Precision@K** |
| :--- | :---: | :---: | :---: |
| **Top 10 bài tiềm năng nhất** | 10 | 10 | **100.0%** |
| **Top 20 bài tiềm năng nhất** | 20 | 20 | **100.0%** |
| **Top 50 bài tiềm năng nhất** | 50 | 50 | **100.0%** |
| **Top 100 bài tiềm năng nhất** | 100 | 100 | **100.0%** |
| **Top 500 bài tiềm năng nhất** | 500 | 476 | **95.2%** |
| **Top 1000 bài tiềm năng nhất** | 1,000 | 923 | **92.3%** |
