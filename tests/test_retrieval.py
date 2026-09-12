"""
Unit tests for CaseRetriever (Sparse, Dense, and Hybrid RRF modes).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import get_or_build_retriever


class TestHybridRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = get_or_build_retriever(load_dense=True)

    def test_sparse_retrieval_returns_cases(self):
        """Sparse retrieval returns expected dictionary format and valid scores."""
        query = "Where is my package? Order was delayed."
        cases = self.retriever.retrieve(query, k=3, mode="sparse")
        self.assertEqual(len(cases), 3)
        for c in cases:
            self.assertIn("conversation_id", c)
            self.assertIn("similarity_score", c)
            self.assertIn("customer_text", c)
            self.assertIn("brand_reply_text", c)
            self.assertEqual(c["retrieval_mode"], "sparse")

    def test_dense_retrieval_returns_cases(self):
        """Dense semantic retrieval operates cleanly with all-MiniLM-L6-v2."""
        query = "where is my stuff"
        cases = self.retriever.retrieve(query, k=3, mode="dense")
        self.assertGreaterEqual(len(cases), 1)
        self.assertEqual(cases[0]["retrieval_mode"], "dense")
        self.assertGreater(cases[0]["similarity_score"], 0.0)

    def test_hybrid_rrf_fusion(self):
        """Hybrid mode produces calibrated composite similarity and RRF scores."""
        query = "I was double billed for Prime Video subscription."
        cases = self.retriever.retrieve(query, k=3, mode="hybrid")
        self.assertEqual(len(cases), 3)
        top_case = cases[0]
        self.assertEqual(top_case["retrieval_mode"], "hybrid")
        self.assertIn("rrf_score", top_case)
        self.assertIn("dense_score", top_case)
        self.assertIn("sparse_score", top_case)
        self.assertGreater(top_case["rrf_score"], 0.0)

    def test_paraphrase_grounding_improvement(self):
        """
        Conversational paraphrase with zero exact keywords ('where is my stuff')
        must yield higher grounding similarity in Hybrid mode than in Sparse mode.
        """
        query = "where is my stuff"
        sparse_res = self.retriever.retrieve(query, k=1, mode="sparse")
        hybrid_res = self.retriever.retrieve(query, k=1, mode="hybrid")

        sparse_sim = sparse_res[0]["similarity_score"]
        hybrid_sim = hybrid_res[0]["similarity_score"]

        self.assertGreater(
            hybrid_sim,
            sparse_sim,
            f"Hybrid sim ({hybrid_sim}) should exceed Sparse sim ({sparse_sim}) on colloquial paraphrasing."
        )


if __name__ == "__main__":
    unittest.main()
