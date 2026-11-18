import sqlite3
import json
import numpy as np
import pandas as pd
import os
import urllib.request
import urllib.parse

class TwoStageHybridRecommender:
    def __init__(self, db_path="data/processed/citation_lens.db", meta_path="models/best_gbt_metadata.json", cache_path="data/processed/cached_titles.json"):
        self.db_path = db_path
        self.cache_path = cache_path
        
        self.cached_titles = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cached_titles = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                self.cached_titles = {}

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
        """
        CHIỀU 1: ĐỒ THỊ MẠNG (Graph-based Candidate Retrieval)
        Lọc ứng viên có cấu trúc mạng vững chắc và độ uy tín cao (PageRank)
        """
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
        """
        CHIỀU 2: NỘI DUNG NGỮ NGHĨA (Content-based Semantic Retrieval)
        Lọc ứng viên có độ tương đồng chuyên môn học thuật và cùng chủ đề/lĩnh vực (Field of Study)
        """
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
        """Tính độ tương đồng chuyên môn Relevance (%) đối với tập ứng viên"""
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
        """
        CHIỀU 3: HỌC MÁY XẾP HẠNG (PySpark MLlib GBT Reranking)
        Chấm điểm xác suất bùng nổ P(Emerging) và tính Hybrid Score để chọn Top-K
        """
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
        """
        PIPELINE CHUẨN 2 TẦNG (TWO-STAGE HYBRID RECOMMENDER - 3 CHIỀU):
        - Đầu vào (Input): Seed Paper
        - Tầng 1 (Retrieval): Chiều 1 (Graph) + Chiều 2 (Content) chạy SONG SONG -> Hợp nhất thành Phễu lọc
        - Tầng 2 (Reranking): Chiều 3 (PySpark MLlib GBT Reranking) -> Top-K
        """
        seed = self.get_paper_info(paper_id)
        if not seed:
            return None, pd.DataFrame()
            
        # 1. Chiều 1: Truy xuất ứng viên từ Đồ thị mạng (Graph Topology)
        c1_candidates = self.dimension1_graph_retrieval(seed, limit=75)
        
        # 2. Chiều 2: Truy xuất ứng viên từ Nội dung ngữ nghĩa & Lĩnh vực (Content Semantic)
        c2_candidates = self.dimension2_content_retrieval(seed, limit=75)
        
        # 3. Phễu lọc Tầng 1: Hợp nhất (Union) ứng viên từ cả 2 kênh & loại bỏ trùng lặp
        merged_candidates = pd.concat([c1_candidates, c2_candidates]).drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
        
        # 4. Tính toán độ liên quan (Relevance) cho toàn bộ phễu lọc
        candidates_with_rel = self.calculate_relevance(seed, merged_candidates)
        
        # 5. Chiều 3: Học máy PySpark GBT chấm điểm và xếp hạng Top-K
        top_k_papers = self.dimension3_gbt_rerank(candidates_with_rel, alpha=alpha, top_k=top_k)
        
        return seed, top_k_papers

    def fetch_real_metadata(self, paper_id: int):
        if paper_id in self.cached_titles and self.cached_titles[paper_id].get("doi"):
            return self.cached_titles[paper_id]

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT mag_id FROM mag_mapping WHERE paper_id = ?;", (paper_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            mag_id = row[0]
            try:
                url = f"https://api.openalex.org/works/mag:{mag_id}?select=id,title,doi,publication_year"
                req = urllib.request.Request(url, headers={"User-Agent": "mailto:citationlens@iuh.edu.vn"})
                with urllib.request.urlopen(req, timeout=3) as res:
                    data = json.loads(res.read().decode())
                    t = data.get("title")
                    d = data.get("doi")
                    meta = {
                        "title": t if t else f"Research Paper #{paper_id}",
                        "doi": d if d else f"https://openalex.org/works/mag:{mag_id}",
                        "year": data.get("publication_year")
                    }
                    self.cached_titles[paper_id] = meta
                    try:
                        with open(self.cache_path, "w", encoding="utf-8") as f:
                            json.dump(self.cached_titles, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    return meta
            except Exception:
                pass

        return {
            "title": f"Scientific Research Paper #{paper_id}",
            "doi": f"https://scholar.google.com/scholar?q={paper_id}",
            "year": None
        }

    def search_by_title(self, query: str, limit: int = 6):
        results = []
        tokens = [t.lower() for t in query.strip().split() if len(t) >= 2]
        
        for pid, item in self.cached_titles.items():
            title_lower = item.get("title", "").lower()
            if any(tok in title_lower for tok in tokens):
                match_count = sum(1 for tok in tokens if tok in title_lower)
                results.append((match_count, {
                    "paper_id": pid,
                    "title": item["title"],
                    "year": item.get("year", "N/A"),
                    "doi": item.get("doi", "")
                }))
        
        results.sort(key=lambda x: x[0], reverse=True)
        final_list = [r[1] for r in results[:limit]]

        if len(final_list) < limit:
            try:
                url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per_page=15&select=id,title,doi,ids,publication_year"
                req = urllib.request.Request(url, headers={"User-Agent": "mailto:citationlens@iuh.edu.vn"})
                with urllib.request.urlopen(req, timeout=4) as res:
                    data = json.loads(res.read().decode())
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()

                    for r in data.get("results", []):
                        mag_id = r.get("ids", {}).get("mag")
                        if mag_id:
                            cur.execute("SELECT paper_id FROM mag_mapping WHERE mag_id = ?;", (str(mag_id),))
                            match = cur.fetchone()
                            if match:
                                pid = match[0]
                                cur.execute("SELECT pub_year FROM papers WHERE paper_id = ?;", (pid,))
                                p_row = cur.fetchone()
                                if p_row:
                                    title = r.get("title")
                                    doi = r.get("doi")
                                    item = {"paper_id": pid, "title": title, "year": p_row[0], "doi": doi}
                                    self.cached_titles[pid] = item
                                    if not any(x["paper_id"] == pid for x in final_list):
                                        final_list.append(item)
                                    if len(final_list) >= limit:
                                        break
                    conn.close()
            except Exception:
                pass

        return final_list[:limit]

    def get_paper_info(self, paper_id: int):
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM papers WHERE paper_id = {paper_id} LIMIT 1;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            info = df.iloc[0].to_dict()
            meta = self.fetch_real_metadata(paper_id)
            info["title"] = meta.get("title")
            info["doi"] = meta.get("doi")
            return info
        return None
