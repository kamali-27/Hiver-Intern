"""
Unit tests for text preprocessing, normalization, and regex rules.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import clean_text
from src.intent_rules import match_intent_rules


class TestDataPreprocessing(unittest.TestCase):

    def test_clean_text_removes_urls(self):
        text = "Check this status https://amzn.to/3xyz now!"
        cleaned = clean_text(text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("amzn.to", cleaned)

    def test_clean_text_removes_twitter_handles(self):
        text = "Hey @AmazonHelp @jeffbezos my package is late!"
        cleaned = clean_text(text)
        self.assertNotIn("@amazonhelp", cleaned)
        self.assertNotIn("@jeffbezos", cleaned)
        self.assertIn("package is late", cleaned)

    def test_clean_text_collapses_whitespace(self):
        text = "Where    is     my     order?   "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "where is my order?")

    def test_clean_text_handles_non_ascii(self):
        text = "Great service 👍💯📦!"
        cleaned = clean_text(text)
        self.assertNotIn("👍", cleaned)
        self.assertIn("great service", cleaned)

    def test_intent_rules_matching(self):
        # Delivery intent
        intent, conf = match_intent_rules("where is my package tracking number")
        self.assertEqual(intent, "order_delivery_delay")
        self.assertGreater(conf, 0.5)

        # Refund intent
        intent, conf = match_intent_rules("i want a full refund for my returned item")
        self.assertEqual(intent, "refund_request")
        self.assertGreater(conf, 0.5)


if __name__ == "__main__":
    unittest.main()
