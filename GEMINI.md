# Workspace Rules: Graph Analytics For Discovering Emerging Influential Papers

## 1. Project Context & Objectives
* **Project Name:** Graph Analytics For Discovering Emerging Influential Papers (Phân tích đồ thị để phát hiện các bài báo tiềm năng bùng nổ).
* **Domain:** Big Data Analytics, Directed Acyclic Graph (DAG) Mining, Bibliometrics, and Machine Learning on Graph Data.
* **Instructor & Academic Context:** Đồ án môn học Big Data - GVHD: TS. Bùi Thanh Hùng (IUH). Template báo cáo: `reference/MauBaocaoBigData.docx`.

---

## 2. Data Architecture & Integrity Constraints
* **Dataset:** OGBN-MAG (694,450 papers).
* **Processed Storage:**
  * Parquet format: `data/processed/citation_ml_dataset.parquet` (Partitioned by `temporal_split`).
  * SQLite database: `data/processed/citation_lens.db` (`papers` table: 694k rows, `top_emerging_papers` table: 44.6k rows).
* **Strict Temporal Splitting (No Data Leakage):**
  * `train`: Papers published $\le 2016$ (~559,373 rows).
  * `valid`: Papers published in $2017$ (~70,198 rows).
  * `test`: Papers published in $2018$ and evaluated for citation growth $\ge 2019$ (~64,879 rows).
  * **Rule:** Never use standard random K-Fold cross-validation across the whole dataset. Temporal ordering must be strictly respected to prevent future citations from leaking into past predictions.

---

## 3. Feature Engineering & Multicollinearity Rules
* **Core 3 Features:**
  * `citation_velocity` ($= \frac{\text{past\_in\_degree}}{\text{paper\_age}}$): Vận tốc trích dẫn tích lũy.
  * `pagerank_score`: Điểm thẩm quyền cấu trúc đồ thị (tính từ GraphFrames).
  * `paper_age`: Số năm bài báo đã tồn tại ($2018 - \text{pub\_year} + 1$).
* **Multicollinearity Prevention:**
  * Do NOT add `past_in_degree` directly in raw linear form alongside `citation_velocity` ($\rho = 0.923 \ge 0.8$, causing severe multicollinearity).
  * If using citation volume, use non-linear transformations (e.g., Logarithmic elasticity or prestige ratios like `prestige_ratio = pagerank_score / log1p(past_in_degree + 1.0)`).

---

## 4. Modeling & Machine Learning Guidelines
* **Distributed ML Framework:** PySpark MLlib (`GBTClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier`, `LinearSVC`, `LogisticRegression`).
* **Class Imbalance Strategy:**
  * Data ratio is ~1 : 6.4 (13.5% positive labels).
  * **Proposed Primary Method:** **Class Weight** (`weightCol="class_weight"`) using inverse class frequencies ($w_0 \approx 0.53, w_1 \approx 9.64$).
  * Secondary baseline: Random Undersampling (RUS) for high-recall scenarios.
* **GBT Hyperparameter Tuning Standard:**
  * Use `ParamGridBuilder` evaluated on `valid` split with `areaUnderPR`.
  * Optimal baseline: `maxDepth=4, maxIter=15` (prevents heavy-tail overfitting and reduces training time by 20%).

---

## 5. Evaluation & Reporting Standards
* **Do Not Rely on Accuracy Alone:** Accuracy (~85% - 88%) suffers from the Accuracy Paradox due to class imbalance.
* **Mandatory Core Metrics:**
  * Minority Class (Label 1.0): Precision(1), Recall(1), F1-Score(1).
  * Discriminative Metrics: PR-AUC (`areaUnderPR` - most critical for imbalanced data) and ROC-AUC.
* **Top-K Discovery Paradigm:**
  * In production & academic defense, always present **Precision@K** ($K \in [10, 20, 50, 100, 500]$) to demonstrate real-world discovery effectiveness.

---

## 6. Code Cleanliness & Notebook Standards
* **Spark Log Suppression:** Always set `spark.sparkContext.setLogLevel("ERROR")` and `warnings.filterwarnings('ignore')` to keep notebook outputs clean.
* **Resource Cleanup:** Always include `spark.stop()` at the end of execution.
* **No Redundant Prints:** Only output key metrics and final figures; avoid printing internal Spark stage progress lines, raw schemas, or verbose JVM tracebacks.

