"""Tests for the PIT WALL browser interface served by FastAPI."""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
        from degradiq.api import app
    return TestClient(app)


class WebInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _client()

    def test_root_serves_app_html(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("content-type", ""))
        self.assertIn("DEGRADIQ", r.text)
        self.assertIn("PIT WALL", r.text)

    def test_app_alias(self):
        self.assertEqual(self.c.get("/app").status_code, 200)

    def test_assets_served(self):
        css = self.c.get("/assets/styles.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("css", css.headers.get("content-type", ""))
        js = self.c.get("/assets/app.js")
        self.assertEqual(js.status_code, 200)
        self.assertIn("runForecast", js.text)

    def test_api_meta_json(self):
        m = self.c.get("/api/meta").json()
        self.assertEqual(m["service"], "DegradIQ API")
        self.assertEqual(m["app"], "/")

    def test_existing_api_contracts_intact(self):
        self.assertEqual(self.c.get("/health").json()["status"], "ok")
        self.assertEqual(self.c.get("/strategy/first-stop/status").status_code, 200)
        self.assertIn("drivers", self.c.get("/strategy/first-stop/lineup").json())
        # predict still uses the existing schema (unknown driver -> 422)
        self.assertEqual(self.c.post("/strategy/first-stop/predict", json={
            "event_name": "X", "race_laps": 57, "mode": "pre-race",
            "drivers": [{"driver": "ZZZ", "grid_position": 1, "starting_compound": "MEDIUM"}]}).status_code, 422)


class NoFabricationTests(unittest.TestCase):
    def test_client_surfaces_errors_not_placeholder(self):
        src = (WEB / "assets" / "app.js").read_text(encoding="utf-8")
        # On failure the client clears predictions and shows the error; it must not
        # substitute sample/fake forecasts.
        self.assertIn("never fabricate", src.lower())
        self.assertIn("state.predictions = []", src)
        for bad in ("sampleData", "FAKE_", "dummyPredictions", "Math.random"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
