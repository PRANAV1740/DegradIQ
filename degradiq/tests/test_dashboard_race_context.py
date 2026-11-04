"""AppTest smoke test for the offline race-context analysis page."""
from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "dashboard_race_context.py")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


class RaceContextPageTest(unittest.TestCase):
    def test_renders_with_predicted_vs_actual(self):
        at = AppTest.from_file(PAGE, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"Race-context page threw: {at.exception}")
        self.assertEqual(at.title[0].value, "DegradIQ Race Context")
        labels = [m.label for m in at.metric]
        self.assertIn("Pred-vs-actual MAE (in-sample)", labels)

    def test_app_has_race_context_page(self):
        src = Path(APP).read_text(encoding="utf-8")
        self.assertIn("dashboard_race_context.py", src)
        self.assertIn("dashboard_first_stop.py", src)
        self.assertIn("dashboard.py", src)


if __name__ == "__main__":
    unittest.main()
