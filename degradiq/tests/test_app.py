"""AppTest smoke test for the unified multipage Streamlit entry."""
from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class MultipageAppTest(unittest.TestCase):
    def test_entry_renders_default_page(self):
        at = AppTest.from_file(APP, default_timeout=40)
        at.run()
        self.assertFalse(at.exception, f"App entry threw: {at.exception}")
        # Default page is the offline first-stop dashboard.
        self.assertTrue(len(at.title) > 0)
        self.assertEqual(at.title[0].value, "DegradIQ First-Stop")


if __name__ == "__main__":
    unittest.main()
