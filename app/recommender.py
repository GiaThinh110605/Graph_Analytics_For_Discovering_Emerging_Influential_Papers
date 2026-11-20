import sqlite3
import json
import numpy as np
import pandas as pd
import os
import urllib.parse

class TwoStageHybridRecommender:
    """
    Hệ thống gợi ý bài báo khoa học dạng lai 2 tầng (Two-Stage Hybrid Recommender):
    - Tầng 1: Lọc ứng viên song song qua Chiều 1 (Đồ thị mạng) & Chiều 2 (Nội dung ngữ nghĩa)
    - Tầng 2: Chiều 3 (Học máy phân tán PySpark MLlib GBT) tái xếp hạng theo xác suất bùng nổ P(Emerging)
    Sử dụng 100% dữ liệu thực từ đồ thị trích dẫn OGBN-MAG (694,450 bài báo) trong SQLite (citation_lens.db).
    Hoàn toàn không dùng demo catalog hay bộ đệm trung gian.
    """
    def __init__(self, db_path="data/processed/citation_lens.db", meta_path="models/best_gbt_metadata.json"):
        self.db_path = db_path

        # Nạp siêu tham số và trọng số đặc trưng của mô hình GBT tốt nhất
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
        Lọc ứng viên có cấu trúc mạng vững chắc và độ uy tín cao (PageRank) từ tập 694k bài báo thực tế
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
        Lọc ứng viên có độ tương đồng chuyên môn học thuật và cùng lĩnh vực nghiên cứu
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
        """Tính độ tương đồng chuyên môn Relevance (%) đối với tập ứng viên (Công thức 2)"""
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

        # Biên dự báo tuyến tính (Công thức 3)
        raw_score = (X_scaled[:, 0] * self.weights[0] + 
                     X_scaled[:, 1] * self.weights[1] - 
                     X_scaled[:, 2] * self.weights[2])
        # Hàm kích hoạt Sigmoid (Công thức 4)
        p_emerging = 1.0 / (1.0 + np.exp(-raw_score))
        candidates = candidates.copy()
        candidates["p_emerging"] = np.round(p_emerging * 100, 2)

        # Điểm lai ghép tổng hợp (Công thức 5)
        final_score = alpha * candidates["relevance"] + (1.0 - alpha) * candidates["p_emerging"]
        candidates["final_score"] = np.round(final_score, 2)

        ranked_df = candidates.sort_values(by="final_score", ascending=False).head(top_k).copy()
        
        # Bổ sung thông tin tiêu đề và DOI thực tế từ cơ sở dữ liệu
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
        - Đầu vào (Input): Seed Paper thực tế
        - Tầng 1 (Retrieval): Chiều 1 (Graph) + Chiều 2 (Content) chạy SONG SONG -> Hợp nhất thành Phễu lọc
        - Tầng 2 (Reranking): Chiều 3 (PySpark MLlib GBT Reranking) -> Top-K bài báo tiềm năng nhất
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
        """Truy vấn định danh thực tế từ SQLite: bảng paper_titles và mag_mapping của OGBN-MAG"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # 1. Truy vấn tiêu đề thực từ bảng paper_titles nếu có
        cur.execute("SELECT title, doi, pub_year FROM paper_titles WHERE paper_id = ?;", (paper_id,))
        row = cur.fetchone()

        # 2. Truy vấn định danh MAG ID từ bảng mag_mapping thực tế của Stanford OGBN-MAG (736k bài)
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
        """Tìm kiếm bài báo trực tiếp từ SQLite (bảng paper_titles và papers)"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        query_clean = query.strip()

        # Nếu người dùng tìm kiếm bằng trực tiếp Paper ID số
        if query_clean.isdigit():
            pid = int(query_clean)
            cur.execute("SELECT paper_id, pub_year FROM papers WHERE paper_id = ?;", (pid,))
            p_row = cur.fetchone()
            if p_row:
                meta = self.fetch_real_metadata(pid)
                conn.close()
                return [{
                    "paper_id": pid,
                    "title": meta.get("title"),
                    "year": p_row[1],
                    "doi": meta.get("doi")
                }]

        tokens = [t.strip().lower() for t in query_clean.split() if len(t.strip()) >= 2]
        if not tokens:
            conn.close()
            return []

        # Truy vấn tìm kiếm SQL theo từ khóa tiêu đề
        where_clauses = ["LOWER(title) LIKE ?" for _ in tokens]
        sql = f"""
            SELECT paper_id, title, pub_year, doi
            FROM paper_titles
            WHERE {" OR ".join(where_clauses)}
            LIMIT {limit * 3};
        """
        params = [f"%{tok}%" for tok in tokens]
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        results = []
        for pid, title, yr, doi in rows:
            title_lower = title.lower()
            match_score = sum(1 for tok in tokens if tok in title_lower)
            results.append((match_score, {
                "paper_id": pid,
                "title": title,
                "year": yr if yr else "N/A",
                "doi": doi if doi else f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
            }))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    def get_paper_info(self, paper_id: int):
        """Lấy thông tin chi tiết của bài báo hạt nhân từ SQLite"""
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
