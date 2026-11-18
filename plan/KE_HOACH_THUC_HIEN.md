# KẾ HOẠCH THỰC HIỆN ĐỀ TÀI: CITATIONLENS
> **Định hướng điều chỉnh theo góp ý của Giảng viên Hướng dẫn:**
> *"Tập trung vào mô hình, xử lý dữ liệu, cố gắng làm với ít nhất 3 mô hình so sánh đánh giá đa chiều, ứng dụng thiết kế ở mức đơn giản nhất. Dữ liệu dùng các năm trước cho huấn luyện và năm sau cho dự đoán (test)."*

---

## 1. Mục Tiêu Trọng Tâm (Pivot)
* **Giảm tải ứng dụng:** Bỏ qua kiến trúc microservice Docker phức tạp, FastAPI 10 endpoints, RAG LLM. Thay bằng **1 Web Demo tối giản bằng Streamlit** để tra cứu bài báo và xem bảng xếp hạng.
* **Tập trung 90% nỗ lực vào:**
  1. Xử lý dữ liệu & chia mốc thời gian (Temporal Split) chống rò rỉ (Data Leakage).
  2. Thực nghiệm các kỹ thuật cân bằng dữ liệu lệch lớp (Class Imbalance ~6.4%).
  3. Huấn luyện và đối đầu **ít nhất 3 - 4 mô hình** (Heuristic, Linear, Tree-based Boosting, Learning to Rank, Neural Network).
  4. Đánh giá đa chiều (Multi-dimensional Evaluation): PR-AUC, NDCG@K, F1-Score, Temporal Stability, SHAP Explainability.

---

## 2. Kế Hoạch 4 Giai Đoạn Ngắn Gọn

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                        LỘ TRÌNH 4 BƯỚC THỰC HIỆN                       │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────┼────────────────────────────────────┐
  ▼                                    ▼                                    ▼
【Giai đoạn 1】                    【Giai đoạn 2】                      【Giai đoạn 3 & 4】
Chuẩn bị Dữ liệu & Baseline         Huấn luyện 4 Mô hình                 Đánh giá Đa chiều
• Temporal Split (Train/Val/Test)  • Xử lý lệch lớp (Raw/Weight/RUS)   • Xuất 10 biểu đồ
• Baseline: PageRank, EIS          • Train: Logistic, RF, LGBM, LTR     • Dựng Demo Streamlit & Viết Báo cáo
```

### Giai đoạn 1: Xử Lý Dữ Liệu & Thiết Lập Baseline (1 - 2 ngày)
1. **Thiết lập Temporal Split nghiêm ngặt** từ tập `citation_ml_dataset.parquet`:
   * **Train Split ($\le 2016$):** 559.373 bài báo.
   * **Validation Split ($2017$):** 70.198 bài báo (tune hyperparameters & thresholds).
   * **Test Split ($2018$):** 64.879 bài báo (kiểm thử khả năng dự báo tương lai).
2. **Xây dựng Heuristic Baselines:**
   * Baseline 1: In-degree (Citation Count tích lũy).
   * Baseline 2: Spark PageRank Score.
   * Baseline 3: EIS Score (Emerging Influence Score do nhóm đề xuất kết hợp Velocity & Age penalty).
   * Đo đạc điểm cơ sở: NDCG@10, PR-AUC.

### Giai đoạn 2: Huấn Luyện 4 Mô Hình & Thực Nghiệm Cân Bằng Dữ Liệu (2 - 3 ngày)
1. **Bộ 4 mô hình so sánh đối đầu:**
   * **M1 - Logistic Regression:** Baseline tuyến tính thống kê.
   * **M2 - Random Forest:** Đại diện Bagging ensemble.
   * **M3 - LightGBM Classifier:** Đại diện Gradient Boosting (SOTA tabular data).
   * **M4 - LightGBM Ranker (LambdaMART):** Đại diện Learning-to-Rank, tối ưu trực tiếp NDCG@10.
   * *(Tùy chọn mở rộng: MLP / GraphSAGE tận dụng vector 128 chiều).*
2. **3 Chiến lược cân bằng dữ liệu (Dựa trên He & Garcia, 2009; Chawla et al., 2002):**
   * **C0 (Raw Imbalanced):** Giữ nguyên tỷ lệ 6.4% nhãn dương.
   * **C1 (Algorithm-level):** Cost-Sensitive Learning / `scale_pos_weight = 14.6` (phạt nặng khi đoán sót nhãn 1).
   * **C2 (Data-level - RUS):** Random Under-Sampling lớp 0 trên tập Train về tỷ lệ 1:3.
   * **C3 (Data-level - SMOTE):** Sinh mẫu nhân tạo cho lớp 1.
   *(Lưu ý: Chỉ cân bằng trên tập Train, giữ nguyên phân bố tự nhiên trên tập Val & Test).*

### Giai đoạn 3: Đánh Giá Đa Chiều & Trực Quan Hóa (2 ngày)
1. **Tổng hợp Bảng Kết Quả Thực Nghiệm (Experiment Matrix):**
   * Đo lường trên 5 chỉ số: `PR-AUC`, `ROC-AUC`, `F1-Score`, `Precision@10`, `NDCG@10`.
2. **Xuất 5 Nhóm Biểu Đồ Trực Quan Bắt Buộc Cho Báo Cáo:**
   * **B1 (Model Comparison):** Grouped Bar Chart so sánh 4 mô hình trên các metric.
   * **B2 (Imbalance Impact):** Heatmap so sánh trước vs sau khi áp dụng các kỹ thuật cân bằng dữ liệu.
   * **B3 (Imbalance Curve):** Precision-Recall Curve (kèm baseline nét đứt 0.064).
   * **B4 (Ranking Curve):** Precision@K và NDCG@K với $K \in [5, 10, 20, 50, 100]$.
   * **B5 (Explainable AI):** SHAP Summary Plot giải thích vai trò của PageRank, Velocity, Paper Age.
   * **B6 (Temporal Robustness):** Biểu đồ F1-score theo từng nhóm tuổi bài báo (`paper_age`: 1-5 năm).

### Giai đoạn 4: Xây Dựng Demo Tối Giản & Hoàn Thiện Báo Cáo (1 - 2 ngày)
1. **Xây dựng Streamlit Minimal Demo (`app.py` ~150 dòng):**
   * **Tab 1 - Leaderboard Top Emerging Papers:** Lọc theo năm hoặc chủ đề, hiển thị danh sách Top 10 bài bùng nổ nhất kèm điểm số của LightGBM Ranker và EIS.
   * **Tab 2 - Paper Inspector:** Nhập `paper_id` $\rightarrow$ Hiển thị xác suất bùng nổ $P(\text{emerging})$, các chỉ số đồ thị và giải thích nhanh lý do (SHAP waterfall).
2. **Hoàn thiện Báo cáo Cuối Kỳ (Theo mẫu `reference/MauBaocaoBigData.docx`):**
   * Đưa bảng kết quả đối đầu và các biểu đồ thực nghiệm vào chương 3 & 4.
   * Kết luận rõ ràng về phương pháp cân bằng dữ liệu và mô hình tối ưu nhất.

---

## 3. Ma Trận Đánh Giá Đa Chiều Mẫu

| Mô hình | Kỹ thuật cân bằng | PR-AUC (Chính) | ROC-AUC | F1-Score | Recall (Nhãn 1) | NDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Citation Count (Heuristic)** | Không áp dụng | 0.12 | 0.61 | 0.18 | 21.0% | 0.35 |
| **PageRank (Heuristic)** | Không áp dụng | 0.15 | 0.65 | 0.22 | 26.5% | 0.41 |
| **EIS Score (Đề xuất)** | Heuristic Đa chỉ số | 0.28 | 0.74 | 0.36 | 45.0% | 0.58 |
| **Logistic Regression** | Gốc (Raw)<br>Class Weight | ...<br>... | ...<br>... | ...<br>... | ...<br>... | ...<br>... |
| **Random Forest** | Gốc (Raw)<br>Class Weight | ...<br>... | ...<br>... | ...<br>... | ...<br>... | ...<br>... |
| **LightGBM Classifier** | Gốc (Raw)<br>`scale_pos_weight`<br>RUS<br>SMOTE | ...<br>...<br>...<br>... | ...<br>...<br>...<br>... | ...<br>...<br>...<br>... | ...<br>...<br>...<br>... | ...<br>...<br>...<br>... |
| **LightGBM Ranker (LambdaMART)** | LTR Grouped by Year | **Đỉnh cao** | - | - | - | **Tối ưu nhất** |

---

## 4. Danh Mục Sản Phẩm Bàn Giao (Deliverables)
1. **Mã nguồn Xử lý dữ liệu & Huấn luyện:** Notebook [`analysis-dataset-after-handle.ipynb`](file:///Users/mac/Downloads/Graph_Analytics_For_Discovering_Emerging_Influential_Papers/analysis-dataset/analysis-dataset-after-handle.ipynb) hoàn chỉnh, có markdown giải thích từng bước.
2. **Bộ biểu đồ chất lượng cao:** File ảnh PNG (DPI 300) phục vụ chèn vào báo cáo và slide.
3. **Ứng dụng Demo tối giản:** File `app.py` chạy bằng lệnh `streamlit run app.py`.
4. **Báo cáo đồ án:** File Word theo mẫu chuẩn của khoa, tập trung 80% dung lượng vào quy trình dữ liệu, thuật toán và phân tích kết quả thực nghiệm.

