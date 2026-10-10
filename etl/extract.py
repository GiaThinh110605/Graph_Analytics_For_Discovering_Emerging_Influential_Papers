from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

# 1. Class Extractor
class Extractor:
    def __init__(self, spark: SparkSession, base_path: str = ""):
        self.spark = spark
        self.base_path = base_path

    # Hàm đọc CSV dùng chung
    def read_csv(self, relative_path: str, columns):
        path = os.path.join(self.base_path, relative_path)
        df = self.spark.read.csv(path, header=False, inferSchema=False)
        return df.toDF(*columns)

    # 2. Đọc năm xuất bản (thêm paper_id theo thứ tự dòng)
    def extract_years(self):
        df = self.read_csv("raw/node-feat/paper/node_year.csv.gz", ["pub_year"])
        return (
            df.rdd.zipWithIndex().map(
                lambda x: (int(x[0]["pub_year"]), int(x[1]))
            ).toDF(["pub_year", "paper_id"])
        )

    # 3. Đọc quan hệ trích dẫn
    def extract_citations(self):
        df = self.read_csv("raw/relations/paper___cites___paper/edge.csv.gz", ["src", "dst"])
        return df.select(
            F.col("src").cast("long"),
            F.col("dst").cast("long")
        )

    # 4. Đọc quan hệ tác giả - bài báo
    def extract_authors(self):
        df = self.read_csv("raw/relations/author___writes___paper/edge.csv.gz", ["author_id", "paper_id"])
        return df.select(
            F.col("author_id").cast("long"),
            F.col("paper_id").cast("long")
        )

    # 5. Đọc quan hệ bài báo - chủ đề
    def extract_topics(self):
        df = self.read_csv("raw/relations/paper___has_topic___field_of_study/edge.csv.gz", ["paper_id", "topic_id"])
        return df.select(
            F.col("paper_id").cast("long"),
            F.col("topic_id").cast("long")
        )

    # 6. Đọc vector đặc trưng 128 chiều (thêm paper_id theo thứ tự dòng)
    def extract_features(self):
        columns = [f"f{i}" for i in range(128)]
        df = self.read_csv("raw/node-feat/paper/node-feat.csv.gz", columns)
        return (
            df.rdd.zipWithIndex().map(lambda x: (tuple(float(val) for val in x[0]) + (int(x[1]),)))
            .toDF(columns + ["paper_id"])
        )

    # 7. Đọc tập train, validation, test (ép kiểu paper_id sang long)
    def extract_splits(self):
        train = self.read_csv("split/time/paper/train.csv.gz", ["paper_id"]).withColumn("split_group", F.lit("train"))
        valid = self.read_csv("split/time/paper/valid.csv.gz", ["paper_id"]).withColumn("split_group", F.lit("valid"))
        test  = self.read_csv("split/time/paper/test.csv.gz",  ["paper_id"]).withColumn("split_group", F.lit("test"))
        
        all_splits = train.unionByName(valid).unionByName(test)
        return all_splits.withColumn("paper_id", F.col("paper_id").cast("long"))


# 8. Chạy thử toàn bộ Extract
if __name__ == "__main__":
    BASE_PATH = (
        "/Users/mac/Downloads/"
        "Graph_Analytics_For_Discovering_Emerging_Influential_Papers/"
        "data/mag"
    )

    spark = SparkSession.builder.appName("CitationLens-Extract").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    ex = Extractor(spark, BASE_PATH)

    datasets = {
        "Paper Years": ex.extract_years(),
        "Citations": ex.extract_citations(),
        "Authors": ex.extract_authors(),
        "Topics": ex.extract_topics(),
        "Features": ex.extract_features(),
        "Splits": ex.extract_splits()
    }

    for name, df in datasets.items():
        df.printSchema()
        df.show(3)

    spark.stop()

