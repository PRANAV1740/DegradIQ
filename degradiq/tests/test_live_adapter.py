"""Tests for the live/replay data-source adapter (Phase VI).

Verifies the honest-unavailable default, point-in-time replay (no future leakage),
status/freshness reporting, and that scoring never fabricates rows.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from degradiq.pit_strategy import FEATURE_COLUMNS, MODEL_PATH
from degradiq.prediction_engine import PredictionEngine
from degradiq.live_adapter import (
    UnavailableLiveSource, ReplayLiveSource, refresh_prediction, LiveSource,
)

FIXTURE = (Path(__file__).resolve().parents[1] / "reports" / "experiments"
           / "first_stop_v2" / "replay_sample_spanish_2024.csv")
PROD = Path(MODEL_PATH).exists()


class UnavailableSourceTests(unittest.TestCase):
    def test_reports_unavailable_and_no_rows(self):
        src = UnavailableLiveSource()
        st = src.status()
        self.assertFalse(st.available)
        self.assertEqual(st.source, "unavailable")
        self.assertTrue(st.reason)                 # must explain why
        self.assertIsNone(src.feature_rows())      # never fabricates data
        self.assertIsInstance(src, LiveSource)


class ReplaySourceTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(FIXTURE)
        self.src = ReplayLiveSource(self.df)

    def test_missing_columns_raises(self):
        with self.assertRaises(ValueError):
            ReplayLiveSource(pd.DataFrame({"LapNumber": [1]}))

    def test_point_in_time_never_returns_future_laps(self):
        self.src.seek(10)
        rows = self.src.feature_rows()
        self.assertTrue((rows["LapNumber"].astype(int) <= 10).all())
        st = self.src.status()
        self.assertTrue(st.available)
        self.assertEqual(st.source, "historical_replay")
        self.assertEqual(st.current_lap, 10)
        self.assertGreater(st.total_laps, 10)

    def test_advance_and_finish(self):
        self.src.seek(self.src._min_lap)
        self.assertFalse(self.src.finished)
        self.src.seek(self.src._max_lap)
        self.assertTrue(self.src.finished)

    def test_actual_first_stop_lookup(self):
        drv = str(self.df["Driver"].iloc[0])
        lap = self.src.actual_first_stop_lap(drv)
        self.assertTrue(lap is None or isinstance(lap, int))


@unittest.skipUnless(PROD, "production model artifact not present")
class RefreshTests(unittest.TestCase):
    def test_refresh_scores_available_rows_only(self):
        engine = PredictionEngine(MODEL_PATH)
        src = ReplayLiveSource(pd.read_csv(FIXTURE)).seek(8)
        out = refresh_prediction(engine, src)
        self.assertTrue(out["status"].available)
        self.assertIn("hazard", out["scored"].columns)
        self.assertTrue(((out["scored"]["hazard"] >= 0) & (out["scored"]["hazard"] <= 1)).all())

    def test_refresh_unavailable_source_scores_nothing(self):
        engine = PredictionEngine(MODEL_PATH)
        out = refresh_prediction(engine, UnavailableLiveSource())
        self.assertFalse(out["status"].available)
        self.assertIsNone(out["scored"])   # no fabricated predictions


if __name__ == "__main__":
    unittest.main()
