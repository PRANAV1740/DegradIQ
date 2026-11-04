"""Tests for the Phase III timing-model comparison harness."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from degradiq.scripts import timing_model_comparison as tmc


def _rows(event, driver, season, stop_lap, n=20):
    rows = []
    for lap in range(1, n + 1):
        rows.append({
            "EventName": event, "Driver": driver, "Season": season,
            "LapNumber": float(lap), "RaceLaps": float(n),
            "GridPosition": 5.0, "TyreLife": float(lap), "Compound": "MEDIUM",
            "Team": "T", "StopNextLap": 1 if lap == stop_lap - 1 else 0,
        })
    return rows


class DriverTableTests(unittest.TestCase):
    def test_same_driver_circuit_distinct_per_season(self):
        # The same driver + circuit in two seasons must be TWO driver-stops, not one.
        df = pd.DataFrame(_rows("GP", "VER", 2022, 10) + _rows("GP", "VER", 2023, 14))
        tab = tmc.driver_table(df)
        self.assertEqual(len(tab), 2)
        self.assertEqual(sorted(tab["StopLap"]), [10, 14])
        self.assertEqual(sorted(tab["Season"]), [2022, 2023])

    def test_driver_without_stop_excluded(self):
        rows = _rows("GP", "NOR", 2022, 10)
        for r in rows:
            r["StopNextLap"] = 0  # never stops
        tab = tmc.driver_table(pd.DataFrame(rows))
        self.assertTrue(tab.empty)


class QuantileTests(unittest.TestCase):
    def test_quantile_predictions_are_monotonic(self):
        frames = []
        for s in (2022, 2023):
            for d in range(12):
                frames += _rows(f"GP{d % 4}", f"D{d}", s, 8 + (d % 10))
        tab = tmc.driver_table(pd.DataFrame(frames))
        models, cats = tmc._fit_quantiles(tab, tmc.CAT_FULL)
        preds = tmc._predict_quantiles(models, cats, tab)
        self.assertEqual(preds.shape[1], 3)
        self.assertTrue((preds[:, 0] <= preds[:, 1] + 1e-9).all())
        self.assertTrue((preds[:, 1] <= preds[:, 2] + 1e-9).all())

    def test_metrics_shape(self):
        m = tmc._metrics(np.array([5, 6]), np.array([8, 9]), np.array([12, 13]),
                         np.array([9, 9]), np.array([10, 10]))
        for k in ("mae", "baseline_mae", "interval_coverage_10_90", "interval_width_10_90_mean"):
            self.assertIn(k, m)
        self.assertEqual(m["n"], 2)


if __name__ == "__main__":
    unittest.main()
