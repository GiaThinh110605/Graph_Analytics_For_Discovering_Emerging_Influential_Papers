import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from graphframes import GraphFrame
from .extract import Extractor

class Transformer:
    def __init__(self, spark: SparkSession, extracted: dict):
        self.spark = spark
        self.years_df = extracted["Paper Years"]
        self.cites_df = extracted["Citations"]
        self.authors_df = extracted["Authors"]
        self.topics_df = extracted["Topics"]
        self.feats_df = extracted["Features"]
        self.splits_df = extracted["Splits"]

    def split_edges_by_time(self):
        cites_timed = self.cites_df.join(
            self.years_df.select(F.col("paper_id").alias("src"), F.col("pub_year").alias("src_year")),
            on="src",
            how="inner"
        )
        edges_past = cites_timed.filter(F.col("src_year") <= 2018)
        edges_future = cites_timed.filter(F.col("src_year") == 2019)
        return edges_past, edges_future

    def compute_pagerank(self, edges_past: DataFrame) -> DataFrame:
        vertices = self.years_df.select(F.col("paper_id").alias("id"))
        edges = edges_past.select("src", "dst")

        g = GraphFrame(vertices, edges)
        pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
        return pr_results.vertices.select(
            F.col("id").alias("paper_id"),
            F.col("pagerank").alias("pagerank_score")
        )

    def build_features_X(self, edges_past: DataFrame) -> DataFrame:
        target_papers = self.years_df.filter(F.col("pub_year") <= 2018)

        past_in_deg = edges_past.groupBy("dst").count() \
            .withColumnRenamed("dst", "paper_id") \
            .withColumnRenamed("count", "past_in_degree")

        author_counts = self.authors_df.groupBy("paper_id").count() \
            .withColumnRenamed("count", "num_authors")

        topic_counts = self.topics_df.groupBy("paper_id").count() \
            .withColumnRenamed("count", "num_topics")

        pr_df = self.compute_pagerank(edges_past)

        df_x = target_papers \
            .join(past_in_deg, on="paper_id", how="left").fillna(0, subset=["past_in_degree"]) \
            .join(author_counts, on="paper_id", how="left").fillna(0, subset=["num_authors"]) \
            .join(topic_counts, on="paper_id", how="left").fillna(0, subset=["num_topics"]) \
            .join(pr_df, on="paper_id", how="left").fillna(0.0, subset=["pagerank_score"]) \
            .withColumn("paper_age", F.lit(2018) - F.col("pub_year") + 1) \
            .withColumn("citation_velocity", F.col("past_in_degree") / F.col("paper_age"))

        feature_cols = [f"f{i}" for i in range(128)]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="paper_feat_128d")
        feat_assembled = assembler.transform(self.feats_df).select("paper_id", "paper_feat_128d")

        return df_x.join(feat_assembled, on="paper_id", how="inner")

    def build_labels_Y(self, df_x: DataFrame, edges_future: DataFrame):
        future_gain = edges_future.groupBy("dst").count() \
            .withColumnRenamed("dst", "paper_id") \
            .withColumnRenamed("count", "delta_c")

        df_xy = df_x.join(future_gain, on="paper_id", how="left").fillna(0, subset=["delta_c"])

        df_xy = df_xy.withColumn(
            "temporal_split",
            F.when(F.col("pub_year") <= 2016, "train")
             .when(F.col("pub_year") == 2017, "valid")
             .otherwise("test")
        )

        df_xy = df_xy.join(
            self.splits_df.withColumnRenamed("split_group", "ogb_split"),
            on="paper_id",
            how="left"
        )

        train_only = df_xy.filter(F.col("temporal_split") == "train")
        theta_p95 = train_only.approxQuantile("delta_c", [0.95], 0.0001)[0]
        print(f"Ngưỡng p95 (Train): {theta_p95}")

        df_final = df_xy.withColumn(
            "label",
            F.when(F.col("delta_c") >= theta_p95, 1.0).otherwise(0.0)
        )
        return df_final

    def run_pipeline(self, output_path: str):
        edges_past, edges_future = self.split_edges_by_time()
        df_x = self.build_features_X(edges_past)
        df_final = self.build_labels_Y(df_x, edges_future)

        df_final.write.mode("overwrite") \
            .partitionBy("temporal_split") \
            .parquet(output_path)


if __name__ == "__main__":
    BASE_PATH = "/Users/mac/Downloads/Graph_Analytics_For_Discovering_Emerging_Influential_Papers/data/mag"
    OUTPUT_PARQUET = "/Users/mac/Downloads/Graph_Analytics_For_Discovering_Emerging_Influential_Papers/data/processed/citation_ml_dataset.parquet"

    spark = SparkSession.builder \
        .appName("CitationLens-Transform") \
        .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.12.3") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    extractor = Extractor(spark, BASE_PATH)
    extracted_data = {
        "Paper Years": extractor.extract_years(),
        "Citations": extractor.extract_citations(),
        "Authors": extractor.extract_authors(),
        "Topics": extractor.extract_topics(),
        "Features": extractor.extract_features(),
        "Splits": extractor.extract_splits()
    }

    transformer = Transformer(spark, extracted_data)
    transformer.run_pipeline(OUTPUT_PARQUET)
    print("Chạy hoàn thành transform")
    spark.stop()
