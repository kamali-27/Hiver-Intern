"""
Unit tests for Escalation Logic and Safety Guardrails.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.escalation_logic import EscalationEngine


class TestEscalationLogic(unittest.TestCase):

    def setUp(self):
        self.engine = EscalationEngine(confidence_threshold=0.55, similarity_threshold=0.25)

    def test_explicit_human_trigger_words(self):
        """Messages demanding a supervisor or lawyer must escalate immediately."""
        queries = [
            "Let me speak to a supervisor right now!",
            "I am contacting my lawyer to report fraud.",
            "Transfer me to a human representative immediately."
        ]
        for query in queries:
            result = self.engine.evaluate(
                message=query,
                predicted_intent="general_inquiry",
                intent_confidence=0.90,
                top_retrieval_similarity=0.80
            )
            self.assertEqual(result["decision"], "ESCALATE_TO_HUMAN")
            self.assertTrue(any(f in result["flags"] for f in ["EXPLICIT_HUMAN_REQUEST", "HOSTILE_OR_LEGAL_THREAT"]))

    def test_high_risk_intent_escalation(self):
        """Refund requests and formal complaints must escalate for specialist review."""
        result = self.engine.evaluate(
            message="Please issue my refund for order #12345.",
            predicted_intent="refund_request",
            intent_confidence=0.88,
            top_retrieval_similarity=0.45
        )
        self.assertEqual(result["decision"], "ESCALATE_TO_HUMAN")
        self.assertIn("HIGH_RISK_INTENT:REFUND_REQUEST", result["flags"])

    def test_low_confidence_escalation(self):
        """Predictions below 0.55 confidence must escalate."""
        result = self.engine.evaluate(
            message="Something feels odd with this purchase.",
            predicted_intent="general_inquiry",
            intent_confidence=0.42,
            top_retrieval_similarity=0.35
        )
        self.assertEqual(result["decision"], "ESCALATE_TO_HUMAN")
        self.assertIn("LOW_INTENT_CONFIDENCE", result["flags"])

    def test_low_similarity_grounding_escalation(self):
        """Queries with poor historical precedent similarity (<0.25) must escalate."""
        result = self.engine.evaluate(
            message="Can you ship this via supersonic carrier pigeon?",
            predicted_intent="order_delivery_delay",
            intent_confidence=0.85,
            top_retrieval_similarity=0.12
        )
        self.assertEqual(result["decision"], "ESCALATE_TO_HUMAN")
        self.assertIn("LOW_HISTORICAL_SIMILARITY", result["flags"])

    def test_routine_auto_handle_approval(self):
        """Routine delivery queries with high confidence and precedent similarity should auto-handle."""
        result = self.engine.evaluate(
            message="Where is my order? Tracking says out for delivery today.",
            predicted_intent="order_delivery_delay",
            intent_confidence=0.85,
            top_retrieval_similarity=0.48
        )
        self.assertEqual(result["decision"], "AUTO_HANDLE")
        self.assertEqual(len(result["flags"]), 0)


if __name__ == "__main__":
    unittest.main()
