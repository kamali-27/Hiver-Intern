"""
Historical Case Retrieval Module.

Builds and persists a fast TF-IDF cosine similarity retrieval index over
historical customer support interactions. Given an incoming message, retrieves
the top-k (default k=3) most relevant historical cases containing:
- conversation_id
- historical customer text
- brand's actual past reply
- historical intent
- similarity score (0.0 to 1.0)
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


class CaseRetriever:
    """
    TF-IDF Vector Space Index for Case-Based Customer Support Retrieval.
    """
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self.documents_df: Optional[pd.DataFrame] = None

    def build_index(self, df: pd.DataFrame) -> "CaseRetriever":
        """Fits vectorizer on customer text and stores document representations."""
        print(f"[retrieval] Building index over {len(df):,} historical cases...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_vectors = self.vectorizer.fit_transform(df["customer_text"])
        # Retain necessary fields to keep payload lightweight and fast
        self.documents_df = df[[
            "conversation_id",
            "customer_text",
            "brand_reply_text",
            "intent"
        ]].reset_index(drop=True)
        return self

    def save(self, filepath: str = RETRIEVAL_INDEX_PATH):
        """Saves index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "doc_vectors": self.doc_vectors,
            "documents_df": self.documents_df
        }, filepath)
        print(f"[retrieval] Index saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = RETRIEVAL_INDEX_PATH) -> "CaseRetriever":
        """Loads index from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Retrieval index not found at {filepath}. Please build it first.")
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.doc_vectors = data["doc_vectors"]
        instance.documents_df = data["documents_df"]
        return instance

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical cases most similar to query.
        Returns list of dicts with:
        [conversation_id, customer_text, brand_reply_text, intent, similarity_score]
        """
        if self.vectorizer is None or self.doc_vectors is None:
            raise RuntimeError("Index not initialized or loaded.")

        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Get top-k indices sorted descending
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
                "similarity_score": round(score, 4)
            })

        return results


def get_or_build_retriever() -> CaseRetriever:
    """Helper to load existing retriever or build from processed data if missing."""
    if os.path.exists(RETRIEVAL_INDEX_PATH):
        return CaseRetriever.load(RETRIEVAL_INDEX_PATH)

    if not os.path.exists(LABELED_DATA_PATH):
        from src.derive_intents import derive_and_label_dataset
        df = derive_and_label_dataset()
    else:
        df = pd.read_csv(LABELED_DATA_PATH)

    retriever = CaseRetriever()
    retriever.build_index(df)
    retriever.save(RETRIEVAL_INDEX_PATH)
    return retriever


if __name__ == "__main__":
    retriever = get_or_build_retriever()
    sample_query = "Where is my order? It was supposed to be delivered yesterday!"
    results = retriever.retrieve(sample_query, k=3)
    print(f"\nQuery: {sample_query}\n")
    for i, res in enumerate(results, 1):
        print(f"Top #{i} (Score: {res['similarity_score']:.3f}) [{res['intent']}]:")
        print(f"  Cust: {res['customer_text']}")
        print(f"  Repl: {res['brand_reply_text']}\n")
