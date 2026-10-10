import os
import sqlite3
import pandas as pd
import pyarrow.parquet as pq


class DataLoader:
    def __init__(self, parquet_path: str, db_path: str):
        self.parquet_path = parquet_path
        self.db_path = db_path

    def load_processed_papers(self, batch_size: int = 100000):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS papers;")
        cursor.execute("""
            CREATE TABLE papers (
                paper_id INTEGER PRIMARY KEY,
                pub_year INTEGER,
                past_in_degree INTEGER,
                pagerank_score REAL,
                num_authors INTEGER,
                num_topics INTEGER,
                paper_age INTEGER,
                citation_velocity REAL,
                delta_c INTEGER,
                label REAL,
                temporal_split TEXT,
                ogb_split TEXT
            );
        """)

        dataset = pq.ParquetDataset(self.parquet_path)
        cols_to_read = [
            "paper_id", "pub_year", "past_in_degree", "pagerank_score", "num_authors", 
            "num_topics", "paper_age", "citation_velocity", 
            "delta_c", "label", "temporal_split", "ogb_split"
        ]

        df = dataset.read(columns=cols_to_read).to_pandas()
        df.to_sql("papers", conn, if_exists="append", index=False, chunksize=batch_size)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_label ON papers(label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_pub_year ON papers(pub_year);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_temporal_split ON papers(temporal_split);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_delta_c ON papers(delta_c DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_pagerank ON papers(pagerank_score DESC);")

        conn.commit()
        conn.close()

    def create_top_emerging_view(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS top_emerging_papers;")
        cursor.execute("""
            CREATE TABLE top_emerging_papers AS
            SELECT 
                paper_id,
                pub_year,
                past_in_degree,
                pagerank_score,
                delta_c AS future_citation_gain,
                citation_velocity,
                num_authors,
                num_topics,
                temporal_split
            FROM papers
            WHERE label = 1.0
            ORDER BY delta_c DESC, citation_velocity DESC;
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_top_emerging_id ON top_emerging_papers(paper_id);")
        conn.commit()
        conn.close()

    def run_pipeline(self):
        self.load_processed_papers()
        self.create_top_emerging_view()
        print("Hoàn thành pipeline Load")


if __name__ == "__main__":
    PARQUET_PATH = "/Users/mac/Downloads/Graph_Analytics_For_Discovering_Emerging_Influential_Papers/data/processed/citation_ml_dataset.parquet"
    DB_PATH = "/Users/mac/Downloads/Graph_Analytics_For_Discovering_Emerging_Influential_Papers/data/processed/citation_lens.db"

    loader = DataLoader(parquet_path=PARQUET_PATH, db_path=DB_PATH)
    loader.run_pipeline()
