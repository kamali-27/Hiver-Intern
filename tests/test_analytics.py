"""
Unit tests for the Brand Intelligence & Analytics Engine.
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analytics import BrandAnalyticsEngine, estimate_polarity


class TestBrandAnalytics(unittest.TestCase):

    def setUp(self):
        self.analytics = BrandAnalyticsEngine()

    def test_estimate_polarity(self):
        self.assertLess(estimate_polarity("This is the worst service ever, terrible and ruined."), 0.0)
        self.assertGreater(estimate_polarity("Thank you so much, great service and solved fast!"), 0.0)
        self.assertEqual(estimate_polarity(""), 0.0)

    def test_get_summary_kpis(self):
        kpis = self.analytics.get_summary_kpis()
        self.assertIn("total_historical_cases", kpis)
        self.assertIn("auto_resolution_rate", kpis)
        self.assertIn("human_escalation_rate", kpis)
        self.assertIn("under_escalation_rate", kpis)
        self.assertIn("est_monthly_savings_usd", kpis)
        self.assertGreater(kpis["total_historical_cases"], 100)
        self.assertEqual(kpis["under_escalation_rate"], 0.0)

    def test_get_intent_distribution(self):
        df_intent = self.analytics.get_intent_distribution()
        self.assertIsInstance(df_intent, pd.DataFrame)
        self.assertIn("Intent", df_intent.columns)
        self.assertIn("Ticket Volume", df_intent.columns)
        self.assertIn("Percentage", df_intent.columns)
        self.assertGreater(len(df_intent), 3)

    def test_get_escalation_by_intent(self):
        df_esc = self.analytics.get_escalation_by_intent()
        self.assertIsInstance(df_esc, pd.DataFrame)
        self.assertIn("AUTO_HANDLE", df_esc.columns)
        self.assertIn("ESCALATE_TO_HUMAN", df_esc.columns)
        self.assertGreater(len(df_esc), 3)

    def test_get_friction_keywords(self):
        df_kw = self.analytics.get_friction_keywords()
        self.assertIsInstance(df_kw, pd.DataFrame)
        self.assertIn("Keyword", df_kw.columns)
        self.assertIn("Frequency", df_kw.columns)
        self.assertGreater(len(df_kw), 0)


if __name__ == "__main__":
    unittest.main()
