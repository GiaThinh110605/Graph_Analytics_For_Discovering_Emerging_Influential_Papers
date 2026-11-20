import sqlite3
import json
import numpy as np
import pandas as pd
import os
import urllib.parse

class TwoStageHybridRecommender:
    def __init__(self, db_path="data/processed/citation_lens.db", meta_path="models/best_gbt_metadata.json"):
        self.db_path = db_path

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.scaler_std = np.array(self.metadata["scaler"]["std"])
            self.feature_cols = self.metadata["features"]
            self.weights = np.array([
                self.metadata["feature_importances"].get("citation_velocity", 0.958),
                self.metadata["feature_importances"].get("pagerank_score", 0.022),
                self.metadata["feature_importances"].get("paper_age", 0.020)
            ])
        else:
            self.scaler_std = np.array([3.3556, 3.7158, 1.9620])
            self.feature_cols = ["citation_velocity", "pagerank_score", "paper_age"]
            self.weights = np.array([0.958, 0.022, 0.020])

    def dimension1_graph_retrieval(self, seed_paper: dict, limit: int = 100) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        pub_year = seed_paper["pub_year"]
        paper_id = seed_paper["paper_id"]
        
        query = f"""
            SELECT paper_id, pub_year, past_in_degree, pagerank_score, 
                   paper_age, citation_velocity, label
            FROM papers
            WHERE pub_year BETWEEN {pub_year - 1} AND {pub_year + 1}
              AND paper_id != {paper_id}
            ORDER BY pagerank_score DESC
            LIMIT {limit};
        """
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        return candidates

    def dimension2_content_retrieval(self, seed_paper: dict, limit: int = 100) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        pub_year = seed_paper["pub_year"]
        paper_id = seed_paper["paper_id"]
        num_topics = seed_paper.get("num_topics", 10)
        
        query = f"""
            SELECT paper_id, pub_year, past_in_degree, pagerank_score, 
                   paper_age, citation_velocity, label
            FROM papers
            WHERE pub_year BETWEEN {pub_year - 2} AND {pub_year + 1}
              AND paper_id != {paper_id}
              AND num_topics BETWEEN {max(1, num_topics - 3)} AND {num_topics + 3}
            ORDER BY citation_velocity DESC
            LIMIT {limit};
        """
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        return candidates

    def calculate_relevance(self, seed_paper: dict, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
            
        seed_year = seed_paper["pub_year"]
        seed_pr = seed_paper["pagerank_score"]
        
        year_diff = np.abs(candidates["pub_year"] - seed_year)
        time_sim = 1.0 / (1.0 + year_diff)
        pr_sim = np.minimum(candidates["pagerank_score"], seed_pr) / np.maximum(candidates["pagerank_score"], seed_pr)
        
        relevance = 0.6 * time_sim + 0.4 * pr_sim
        candidates = candidates.copy()
        candidates["relevance"] = np.round(relevance * 100, 2)
        return candidates

    def dimension3_gbt_rerank(self, candidates: pd.DataFrame, alpha: float = 0.4, top_k: int = 10) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()

        X = candidates[self.feature_cols].values
        X_scaled = X / self.scaler_std

        raw_score = (X_scaled[:, 0] * self.weights[0] + 
                     X_scaled[:, 1] * self.weights[1] - 
                     X_scaled[:, 2] * self.weights[2])
        p_emerging = 1.0 / (1.0 + np.exp(-raw_score))
        candidates = candidates.copy()
        candidates["p_emerging"] = np.round(p_emerging * 100, 2)

        final_score = alpha * candidates["relevance"] + (1.0 - alpha) * candidates["p_emerging"]
        candidates["final_score"] = np.round(final_score, 2)

        ranked_df = candidates.sort_values(by="final_score", ascending=False).head(top_k).copy()
        
        titles = []
        dois = []
        for pid in ranked_df["paper_id"]:
            meta = self.fetch_real_metadata(int(pid))
            titles.append(meta.get("title"))
            dois.append(meta.get("doi"))

        ranked_df["title"] = titles
        ranked_df["doi"] = dois
        return ranked_df

    def recommend(self, paper_id: int, alpha: float = 0.4, top_k: int = 10):
        seed = self.get_paper_info(paper_id)
        if not seed:
            return None, pd.DataFrame()
            
        c1_candidates = self.dimension1_graph_retrieval(seed, limit=75)
        c2_candidates = self.dimension2_content_retrieval(seed, limit=75)
        
        merged_candidates = pd.concat([c1_candidates, c2_candidates]).drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
        candidates_with_rel = self.calculate_relevance(seed, merged_candidates)
        top_k_papers = self.dimension3_gbt_rerank(candidates_with_rel, alpha=alpha, top_k=top_k)
        
        return seed, top_k_papers

    def fetch_real_metadata(self, paper_id: int):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT title, doi, pub_year FROM paper_titles WHERE paper_id = ?;", (paper_id,))
        row = cur.fetchone()

        cur.execute("SELECT mag_id FROM mag_mapping WHERE paper_id = ?;", (paper_id,))
        mag_row = cur.fetchone()
        conn.close()

        mag_id = mag_row[0] if mag_row else str(paper_id)
        openalex_url = f"https://openalex.org/works/mag:{mag_id}"

        if row and row[0]:
            return {
                "title": row[0],
                "doi": row[1] if row[1] else openalex_url,
                "year": row[2],
                "mag_id": mag_id
            }

        return {
            "title": f"OGBN-MAG Research Paper #{paper_id}",
            "doi": openalex_url,
            "year": None,
            "mag_id": mag_id
        }

    def search_by_title(self, query: str, limit: int = 6):
        query_clean = query.strip()
        if not query_clean:
            return []

        if query_clean.isdigit():
            pid = int(query_clean)
            info = self.get_paper_info(pid)
            if info:
                return [{
                    "paper_id": pid,
                    "title": info.get("title"),
                    "year": info.get("pub_year"),
                    "doi": info.get("doi")
                }]

        tokens = [t.lower() for t in query_clean.split() if len(t) >= 2]
        if not tokens:
            return []

        where_clause = " OR ".join(["LOWER(title) LIKE ?" for _ in tokens])
        params = [f"%{tok}%" for tok in tokens]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(f"""
                SELECT paper_id, title, pub_year, doi
                FROM paper_titles
                WHERE {where_clause}
                LIMIT {limit * 3}
            """, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            title = row["title"]
            title_lower = title.lower()
            score = sum(1 for tok in tokens if tok in title_lower)
            doi = row["doi"] or f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
            results.append((score, {
                "paper_id": row["paper_id"],
                "title": title,
                "year": row["pub_year"] or "N/A",
                "doi": doi
            }))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    def get_paper_info(self, paper_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM papers WHERE paper_id = ? LIMIT 1;", (paper_id,))
            row = cur.fetchone()

        if not row:
            return None

        info = dict(row)
        meta = self.fetch_real_metadata(paper_id)
        info["title"] = meta.get("title")
        info["doi"] = meta.get("doi")
        return info
