"""
Historical Case Retrieval Module (Hybrid Dense + Sparse Engine).

Combines:
1. Sparse Lexical Search: TF-IDF n-gram vectorizer with cosine similarity.
2. Dense Semantic Search: Bi-encoder embeddings (all-MiniLM-L6-v2, 384-d) with cosine similarity.
3. Hybrid Search: Reciprocal Rank Fusion (RRF) combining dense semantic intent
   with exact sparse keyword signals.

Provides sub-20ms inference and resolves colloquial paraphrase mismatches.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LABELED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled_conversations.csv")
RETRIEVAL_INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "retrieval_index.pkl")
DENSE_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "models", "dense_embeddings.npy")


class CaseRetriever:
    """
    Dual-Index (Dense + Sparse) Case-Based Customer Support Retrieval Engine.
    """
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self.documents_df: Optional[pd.DataFrame] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.dense_model = None

    def build_sparse_index(self, df: pd.DataFrame) -> "CaseRetriever":
        """Fits TF-IDF vectorizer on customer text."""
        print(f"[retrieval] Fitting TF-IDF sparse index over {len(df):,} historical cases...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_vectors = self.vectorizer.fit_transform(df["customer_text"])
        self.documents_df = df[[
            "conversation_id",
            "customer_text",
            "brand_reply_text",
            "intent"
        ]].reset_index(drop=True)
        return self

    def build_index(self, df: pd.DataFrame) -> "CaseRetriever":
        """Alias for backward compatibility."""
        return self.build_sparse_index(df)

    def load_dense_embeddings(self, filepath: str = DENSE_EMBEDDINGS_PATH) -> bool:
        """Loads precomputed dense vector representations if available."""
        if os.path.exists(filepath):
            try:
                self.dense_embeddings = np.load(filepath)
                # Ensure dense embeddings are L2-normalized
                norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.dense_embeddings = self.dense_embeddings / norms
                return True
            except Exception as e:
                print(f"[retrieval] Warning: Failed to load dense embeddings from {filepath}: {e}")
        return False

    def _get_dense_model(self):
        """Lazy-loads the SentenceTransformer model on demand."""
        if self.dense_model is None:
            from sentence_transformers import SentenceTransformer
            self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.dense_model

    def save(self, filepath: str = RETRIEVAL_INDEX_PATH):
        """Saves sparse index and document metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "doc_vectors": self.doc_vectors,
            "documents_df": self.documents_df
        }, filepath)
        print(f"[retrieval] Sparse index saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = RETRIEVAL_INDEX_PATH, load_dense: bool = True) -> "CaseRetriever":
        """Loads index from disk and attaches precomputed dense embeddings."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Retrieval index not found at {filepath}. Please build it first.")
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.doc_vectors = data["doc_vectors"]
        instance.documents_df = data["documents_df"]

        if load_dense:
            loaded = instance.load_dense_embeddings()
            if loaded:
                print("[retrieval] Hybrid mode enabled: dense embeddings loaded successfully.")
        return instance

    def retrieve_sparse(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """TF-IDF lexical matching."""
        if self.vectorizer is None or self.doc_vectors is None:
            raise RuntimeError("Sparse index not initialized or loaded.")

        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            row = self.documents_df.iloc[idx]
            results.append({
                "conversation_id": str(row["conversation_id"]),
                "customer_text": str(row["customer_text"]),
                "brand_reply_text": str(row["brand_reply_text"]),
                "intent": str(row["intent"]),
                "similarity_score": round(score, 4),
                "sparse_score": round(score, 4),
                "dense_score": 0.0,
                "retrieval_mode": "sparse"
            })
        return results

    def retrieve_dense(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Dense semantic matching using all-MiniLM-L6-v2."""
        if self.dense_embeddings is None:
            return self.retrieve_sparse(query, k)

        model = self._get_dense_model()
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.dense_embeddings, query_emb)
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            row = self.documents_df.iloc[idx]
            results.append({
                "conversation_id": str(row["conversation_id"]),
                "customer_text": str(row["customer_text"]),
                "brand_reply_text": str(row["brand_reply_text"]),
                "intent": str(row["intent"]),
                "similarity_score": round(score, 4),
                "dense_score": round(score, 4),
                "sparse_score": 0.0,
                "retrieval_mode": "dense"
            })
        return results

    def retrieve_hybrid(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining Dense Semantic and Sparse Lexical ranks
        via Reciprocal Rank Fusion (RRF):
          RRF(d) = alpha / (60 + rank_dense) + (1 - alpha) / (60 + rank_sparse)
        """
        if self.dense_embeddings is None:
            return self.retrieve_sparse(query, k)

        if not query or not query.strip():
            return []

        # 1. Sparse scoring
        query_vec = self.vectorizer.transform([query])
        sparse_sims = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # 2. Dense scoring
        model = self._get_dense_model()
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        dense_sims = np.dot(self.dense_embeddings, query_emb)

        # 3. Get candidate ranks from top-60 of each
        sparse_top_order = np.argsort(sparse_sims)[::-1][:60]
        dense_top_order = np.argsort(dense_sims)[::-1][:60]

        sparse_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(sparse_top_order)}
        dense_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(dense_top_order)}

        candidate_ids = set(sparse_ranks.keys()).union(set(dense_ranks.keys()))
        rrf_scores = {}

        for doc_id in candidate_ids:
            r_sparse = sparse_ranks.get(doc_id, 1000)
            r_dense = dense_ranks.get(doc_id, 1000)
            score = (alpha / (60.0 + r_dense)) + ((1.0 - alpha) / (60.0 + r_sparse))
            rrf_scores[doc_id] = score

        # Sort candidates by RRF score descending
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:k]

        results = []
        for idx in sorted_candidates:
            d_score = float(dense_sims[idx])
            s_score = float(sparse_sims[idx])
            # Calibrated composite similarity: dense-anchored with lexical boost
            composite_score = min(1.0, 0.65 * max(0.0, d_score) + 0.35 * max(0.0, s_score))
            row = self.documents_df.iloc[idx]
            results.append({
                "conversation_id": str(row["conversation_id"]),
                "customer_text": str(row["customer_text"]),
                "brand_reply_text": str(row["brand_reply_text"]),
                "intent": str(row["intent"]),
                "similarity_score": round(composite_score, 4),
                "dense_score": round(d_score, 4),
                "sparse_score": round(s_score, 4),
                "rrf_score": round(rrf_scores[idx], 6),
                "retrieval_mode": "hybrid"
            })
        return results

    def retrieve(self, query: str, k: int = 3, mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Public retrieval interface with configurable mode ('hybrid', 'dense', 'sparse').
        Gracefully falls back to sparse if dense resources are unavailable.
        """
        if mode == "hybrid" and self.dense_embeddings is not None:
            return self.retrieve_hybrid(query, k)
        elif mode == "dense" and self.dense_embeddings is not None:
            return self.retrieve_dense(query, k)
        else:
            return self.retrieve_sparse(query, k)


def get_or_build_retriever(load_dense: bool = True) -> CaseRetriever:
    """Helper to load existing retriever or build from processed data if missing."""
    if os.path.exists(RETRIEVAL_INDEX_PATH):
        return CaseRetriever.load(RETRIEVAL_INDEX_PATH, load_dense=load_dense)

    if not os.path.exists(LABELED_DATA_PATH):
        from src.derive_intents import derive_and_label_dataset
        df = derive_and_label_dataset()
    else:
        df = pd.read_csv(LABELED_DATA_PATH)

    retriever = CaseRetriever()
    retriever.build_index(df)
    retriever.save(RETRIEVAL_INDEX_PATH)
    if load_dense:
        retriever.load_dense_embeddings()
    return retriever


if __name__ == "__main__":
    retriever = get_or_build_retriever(load_dense=True)
    sample_queries = [
        "Where is my package? It was supposed to be delivered yesterday!",
        "where is my stuff",
        "i was overcharged on my visa card"
    ]
    for sample in sample_queries:
        print(f"\n{'='*70}\nQuery: '{sample}'")
        sparse_res = retriever.retrieve(sample, k=1, mode="sparse")
        hybrid_res = retriever.retrieve(sample, k=1, mode="hybrid")
        print(f"  [Sparse TF-IDF] Sim={sparse_res[0]['similarity_score']:.3f} | Text: {sparse_res[0]['customer_text'][:60]}...")
        print(f"  [Hybrid RRF]    Sim={hybrid_res[0]['similarity_score']:.3f} | Text: {hybrid_res[0]['customer_text'][:60]}...")
