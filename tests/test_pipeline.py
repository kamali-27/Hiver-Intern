"""
End-to-end integration tests for the unified CustomerSupportPipeline.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline


class TestCustomerSupportPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = get_pipeline()

    def test_pipeline_output_contract(self):
        """Verify the pipeline returns all expected dictionary keys and correct types."""
        query = "Where is my package #112-9281726? It was supposed to arrive yesterday."
        result = self.pipeline.process_message(query)

        expected_keys = [
            "message",
            "predicted_intent",
            "intent_confidence",
            "all_intent_probabilities",
            "retrieved_cases",
            "top_retrieval_similarity",
            "generated_reply",
            "grounding_sources",
            "generation_mode",
            "escalation_decision",
            "escalation_reason",
            "escalation_flags",
            "applied_threshold",
            "precedent_boost_applied",
            "risk_tier"
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing expected key: {key}")

        self.assertIsInstance(result["predicted_intent"], str)
        self.assertIsInstance(result["intent_confidence"], float)
        self.assertTrue(0.0 <= result["intent_confidence"] <= 1.0)
        self.assertIn(result["escalation_decision"], ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"])
        self.assertIsInstance(result["retrieved_cases"], list)
        self.assertGreater(len(result["retrieved_cases"]), 0)

    def test_empty_message_handling(self):
        """Empty or whitespace-only messages must be safely handled and escalated."""
        result = self.pipeline.process_message("   ")
        self.assertEqual(result["escalation_decision"], "ESCALATE_TO_HUMAN")
        self.assertIn("EMPTY_MESSAGE", result["escalation_flags"])
        self.assertGreater(len(result["generated_reply"]), 0)

    def test_legal_threat_escalation(self):
        """Hostile messages with legal threats must deterministically escalate."""
        query = "Your company stole my money! I am filing a lawsuit and reporting you to the BBB!"
        result = self.pipeline.process_message(query)
        self.assertEqual(result["escalation_decision"], "ESCALATE_TO_HUMAN")
        self.assertTrue(any("LEGAL" in f or "COMPLAINT" in f for f in result["escalation_flags"]))

    def test_retrieved_cases_structure(self):
        """Retrieved historical cases must contain text, reply, intent, and similarity score."""
        query = "I need to return this damaged shirt."
        result = self.pipeline.process_message(query)
        top_case = result["retrieved_cases"][0]

        for field in ["conversation_id", "customer_text", "brand_reply_text", "intent", "similarity_score"]:
            self.assertIn(field, top_case)
            self.assertIsNotNone(top_case[field])


if __name__ == "__main__":
    unittest.main()
