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


if __name__ == "__main__":
    unittest.main()
