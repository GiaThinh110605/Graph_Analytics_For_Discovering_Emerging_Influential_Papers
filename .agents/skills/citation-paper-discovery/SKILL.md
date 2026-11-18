---
name: citation-paper-discovery
description: >-
  Provides end-to-end workflows and methodologies for analyzing citation networks,
  training PySpark MLlib models to discover emerging influential papers, evaluating
  Top-K ranking metrics, and building the two-stage hybrid recommender system for the CitationLens project.
---

# Citation Paper Discovery Skill

Use this skill when developing, testing, or evaluating graph analytics and machine learning workflows for discovering emerging scientific papers on the OGBN-MAG dataset.

---

## 1. Quick Data & Environment Reference

* **Dataset Path:** `data/processed/citation_ml_dataset.parquet`
* **SQLite Database:** `data/processed/citation_lens.db` (Tables: `papers`, `top_emerging_papers`)
* **Core Features:** `citation_velocity`, `pagerank_score`, `paper_age`
* **Target Label:** `label` (1.0 = Emerging paper in top 5% future citations, 0.0 = Regular paper)
* **Partitions:** `temporal_split` (`train`: $\le 2016$, `valid`: $2017$, `test`: $2018-2019$)

---

## 2. Standard PySpark Session Initialization

When initializing Spark for training or evaluation, use this memory configuration to handle 694k rows efficiently:

```python
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("CitationLens-Discovery") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
```

---

## 3. Workflow A: Running Tuned GBT Classifier with Class Weight

To evaluate the champion model with balanced class weights:

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 1. Assemble and scale
features = ["citation_velocity", "pagerank_score", "paper_age"]
assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
assembled_df = assembler.transform(df)

train_data = assembled_df.filter(F.col("temporal_split") == "train")
test_data = assembled_df.filter(F.col("temporal_split") == "test")

scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(train_data)
train_scaled = scaler_model.transform(train_data)
test_scaled = scaler_model.transform(test_data)

# 2. Compute balanced class weights
counts = {r["label"]: r["count"] for r in train_scaled.groupBy("label").count().collect()}
total = sum(counts.values())
w0 = total / (2 * counts[0.0])
w1 = total / (2 * counts[1.0])
train_weighted = train_scaled.withColumn("class_weight", F.when(F.col("label") == 0.0, w0).otherwise(w1))

# 3. Train optimal GBT model
gbt = GBTClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    weightCol="class_weight",
    maxDepth=4,
    maxIter=15,
    seed=42
)
model = gbt.fit(train_weighted)
predictions = model.transform(test_scaled)
```

---

## 4. Workflow B: Computing Top-K Ranking Metrics (Precision@K)

In research paper discovery and academic defenses, evaluate ranking quality directly on the test set:

```python
import pandas as pd

test_pd = predictions.select("paper_id", "label", "rawPrediction").toPandas()
test_pd["score"] = test_pd["rawPrediction"].apply(lambda v: float(v[1]))
test_ranked = test_pd.sort_values("score", ascending=False)

for k in [10, 20, 50, 100, 500]:
    top_k = test_ranked.head(k)
    prec_k = top_k["label"].mean()
    print(f"Precision@{k}: {prec_k * 100:.2f}% ({int(top_k['label'].sum())}/{k} emerging papers)")
```

*Expected Benchmark:* Precision@10 through @50 $\approx 100\%$, Precision@500 $\ge 95\%$.

---

## 5. Workflow C: Two-Stage Hybrid Recommender Simulation

For demonstrating or implementing the recommendation engine:

1. **Stage 1 (Candidate Retrieval):**
   * Filter candidates by Topic or Co-citation 1-hop / 2-hop graph neighborhood from SQLite `citation_lens.db`.
   * Fast query: `SELECT * FROM top_emerging_papers WHERE pub_year >= 2017 LIMIT 100;`
2. **Stage 2 (ML Reranking):**
   * Score candidates using GBT predicted probability: $P(\text{label}=1)$.
   * Compute final hybrid rank:
     $$\text{Final Score} = 0.4 \times \text{Relevance} + 0.6 \times P(\text{Emerging})$$
   * Return Top 10 emerging gems.

---

## 6. Verification & Rubric Alignment Checklist

Before finalizing any notebook or report section for TS. Bùi Thanh Hùng:
* [ ] No data leakage: temporal split strictly maintained.
* [ ] Multicollinearity check verified: no raw `past_in_degree` placed linearly next to `citation_velocity`.
* [ ] Precision, Recall, F1 for minority class reported alongside PR-AUC and ROC-AUC.
* [ ] Precision@K ranking table included to demonstrate real-world discovery value.
* [ ] Spark session properly terminated (`spark.stop()`).

