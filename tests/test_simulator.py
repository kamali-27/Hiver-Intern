"""
Unit tests for the Multi-Turn Support Simulator Engine.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulator_engine import (
    SimulatorEngine,
    estimate_sentiment,
    determine_frustration_level,
    extract_entities_from_text,
    STRESS_SCENARIOS
)


class TestSimulatorEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SimulatorEngine()

    def test_sentiment_estimation_negative(self):
        text = "Your carrier threw my package into the mud and ruined everything! Terrible and unacceptable."
        sentiment = estimate_sentiment(text)
        self.assertLess(sentiment, -0.4)
        level = determine_frustration_level(sentiment)
        self.assertIn(level, ["FRUSTRATED", "CRITICAL / HOSTILE"])

    def test_sentiment_estimation_positive(self):
        text = "Thank you so much, this is great and very helpful! Problem is solved."
        sentiment = estimate_sentiment(text)
        self.assertGreater(sentiment, 0.4)
        level = determine_frustration_level(sentiment)
        self.assertEqual(level, "SATISFIED / POLITE")

    def test_entity_extraction(self):
        text = "Order #112-9281726-5544332 with tracking TBA9918273645 has a dispute amount of $75.50. Email me at john.doe@example.com"
        entities = extract_entities_from_text(text)

        self.assertIn("order_id", entities)
        self.assertEqual(entities["order_id"], "112-9281726-5544332")
        self.assertIn("tracking_number", entities)
        self.assertEqual(entities["tracking_number"], "TBA9918273645")
        self.assertIn("dispute_amount", entities)
        self.assertEqual(entities["dispute_amount"], "$75.50")
        self.assertIn("customer_email", entities)
        self.assertEqual(entities["customer_email"], "john.doe@example.com")

    def test_simulator_session_creation(self):
        state = self.engine.create_session("late_delivery")
        self.assertTrue(state.conversation_id.startswith("SIM_"))
        self.assertEqual(len(state.turns), 0)
        self.assertEqual(state.current_status, "ACTIVE")

    def test_multi_turn_processing_and_frustration_accumulation(self):
        state = self.engine.create_session("late_delivery")

        # Turn 1: Polite inquiry
        turn1 = self.engine.process_turn(state, "Where is my package #112-9281726? It was supposed to be here yesterday.", retrieval_mode="sparse")
        self.assertEqual(len(state.turns), 1)
        self.assertEqual(turn1.turn_index, 1)
        self.assertIn("order_id", state.extracted_entities)

        # Turn 2: Escalating anger
        turn2 = self.engine.process_turn(state, "It's 3 days late, you ruined my trip, this is terrible!", retrieval_mode="sparse")
        self.assertEqual(len(state.turns), 2)
        self.assertGreater(state.cumulative_frustration, 0.0)

        # Turn 3: Severe hostility
        turn3 = self.engine.process_turn(state, "I want to speak with a supervisor immediately, you thieves stole my money!", retrieval_mode="sparse")
        self.assertEqual(len(state.turns), 3)
        self.assertEqual(state.current_status, "ESCALATED_TO_HUMAN")
        self.assertTrue(state.escalation_triggered)


if __name__ == "__main__":
    unittest.main()
