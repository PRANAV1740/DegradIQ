"""Tests for the Vercel serverless entrypoint, full FastF1 decoupling, and cache resolution."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Package directory and repository root
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent

# Simulate Vercel runtime root directory where `degradiq` is the root
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


class ServerlessDecouplingTest(unittest.TestCase):
    def setUp(self):
        # Explicitly block training-only dependencies to verify runtime isolation
        self._blocked = {}
        for mod in ("fastf1", "fastf1.core", "fastf1.api", "matplotlib", "matplotlib.pyplot"):
            self._blocked[mod] = sys.modules.get(mod)
            sys.modules[mod] = None

    def tearDown(self):
        # Restore blocked modules
        for mod, val in self._blocked.items():
            if val is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = val

    def test_main_app_loads_without_fastf1_or_matplotlib(self):
        """Verify main.app loads cleanly with fastf1 and matplotlib blocked."""
        # Clean existing cached module to test fresh import
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("main", "api"):
                del sys.modules[mod_name]

        import main
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.assertIsInstance(main.app, FastAPI)
        client = TestClient(main.app)

        # 1. Health check
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json(), {"status": "ok", "season": 2026})

        # 2. Calendar route
        res_cal = client.get("/calendar")
        self.assertEqual(res_cal.status_code, 200)
        events = res_cal.json()
        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 23, "Expected exactly 23 official championship rounds.")

        # Round 1
        r1 = events[0]
        self.assertEqual(r1["RoundNumber"], 1)
        self.assertEqual(r1["EventName"], "Australian Grand Prix")
        self.assertIn("Melbourne", r1["Location"])

        # Round 16 (Bahrain rescheduled to Sepang)
        r16 = events[15]
        self.assertEqual(r16["RoundNumber"], 16)
        self.assertEqual(r16["EventName"], "Bahrain Grand Prix")
        self.assertIn("Kuala Lumpur", r16["Location"])

        # Round 17 (Singapore)
        r17 = events[16]
        self.assertEqual(r17["RoundNumber"], 17)
        self.assertEqual(r17["EventName"], "Singapore Grand Prix")
        self.assertIn("sprint", str(r17.get("EventFormat", "")).lower())

        # Round 23 (Abu Dhabi)
        r23 = events[22]
        self.assertEqual(r23["RoundNumber"], 23)
        self.assertEqual(r23["EventName"], "Abu Dhabi Grand Prix")

        # 2b. Calendar metadata file verification
        meta_path = PACKAGE_DIR / "calendar_2026_metadata.json"
        self.assertTrue(meta_path.exists(), "calendar_2026_metadata.json must exist")
        import json
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        self.assertEqual(meta["season"], 2026)
        self.assertEqual(meta["total_rounds"], 23)
        self.assertEqual(meta["total_rounds"], len(events))
        self.assertTrue(meta.get("source"))
        self.assertTrue(meta.get("snapshot_timestamp_utc"))

        # 3. Strategy prediction route contract
        payload = {
            "event_name": "Bahrain Grand Prix",
            "race_laps": 57,
            "mode": "pre-race",
            "drivers": [
                {"driver": "VER", "grid_position": 1, "starting_compound": "SOFT"},
                {"driver": "NOR", "grid_position": 2, "starting_compound": "MEDIUM"},
            ],
        }
        res_pred = client.post("/strategy/first-stop/predict", json=payload)
        self.assertEqual(res_pred.status_code, 200)
        pred_data = res_pred.json()
        self.assertEqual(pred_data["season"], 2026)
        self.assertEqual(pred_data["driver_count_forecast"], 2)
        self.assertEqual(len(pred_data["predictions"]), 2)
        for p in pred_data["predictions"]:
            self.assertIn(p["driver"], ("VER", "NOR"))
            self.assertGreaterEqual(p["probability_stop_before_finish"], 0.0)
            self.assertLessEqual(p["probability_stop_before_finish"], 1.0)
            self.assertTrue(p["lap_probabilities"])
            self.assertTrue(p["cumulative_probabilities"])


class CacheResolutionTest(unittest.TestCase):
    def test_local_cache_resolves_against_project_root(self):
        """Ensure local relative cache paths resolve against PROJECT_ROOT (p:\\DegradIQ)."""
        # Unset VERCEL to test local development resolution
        orig_vercel = os.environ.pop("VERCEL", None)
        try:
            # Need to unblock fastf1 for ingest import in this test
            if sys.modules.get("fastf1") is None:
                sys.modules.pop("fastf1", None)
            from degradiq.ingest import resolve_fastf1_cache_dir

            resolved = resolve_fastf1_cache_dir("data")
            expected = (PROJECT_ROOT / "data").resolve()
            self.assertEqual(resolved, expected)
            self.assertTrue(resolved.exists())
        finally:
            if orig_vercel is not None:
                os.environ["VERCEL"] = orig_vercel

    def test_vercel_cache_resolves_to_tmp(self):
        """Ensure Vercel environment explicitly resolves to /tmp/fastf1_cache."""
        orig_vercel = os.environ.get("VERCEL")
        os.environ["VERCEL"] = "1"
        try:
            if sys.modules.get("fastf1") is None:
                sys.modules.pop("fastf1", None)
            from degradiq.ingest import resolve_fastf1_cache_dir

            resolved = resolve_fastf1_cache_dir("data")
            self.assertEqual(resolved.as_posix(), "/tmp/fastf1_cache")
        finally:
            if orig_vercel is None:
                os.environ.pop("VERCEL", None)
            else:
                os.environ["VERCEL"] = orig_vercel


if __name__ == "__main__":
    unittest.main()
