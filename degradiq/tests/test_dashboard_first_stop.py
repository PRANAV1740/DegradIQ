"""AppTest smoke test for the engine-driven first-stop dashboard (offline)."""
from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard_first_stop.py")


class FirstStopDashboardTest(unittest.TestCase):
    def test_renders_without_exceptions(self):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Dashboard threw: {at.exception}")
        self.assertTrue(len(at.title) > 0)
        self.assertEqual(at.title[0].value, "DegradIQ First-Stop")

    def test_evaluation_table_renders(self):
        # The evaluation section reads the tracked generalisation summary and should
        # render a table in the default (no-interaction) run.
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(len(at.subheader) >= 1)

    def test_multi_event_replay_fixture(self):
        import pandas as pd
        fx = (Path(__file__).resolve().parents[1] / "reports" / "experiments"
              / "first_stop_v2" / "replay_events_2024.csv")
        self.assertTrue(fx.exists(), "multi-event replay fixture missing")
        df = pd.read_csv(fx)
        self.assertGreater(df["EventName"].nunique(), 1)  # several replayable events
        self.assertIn("StopNextLap", df.columns)


if __name__ == "__main__":
    unittest.main()
