"""Synthetic smoke tests for the first-stop hazard model; no real F1 result is implied."""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd

from degradiq.config import FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
from degradiq.pit_strategy import build_hazard_rows, fit_first_stop_model, predict_first_stops


class FirstStopModelSmokeTest(unittest.TestCase):
    def make_laps(self):
        drivers = sorted(FULL_SEASON_DRIVERS)
        rows = []
        for index, driver in enumerate(drivers):
            stop_lap = 13 + (index % 8)
            for lap in range(1, 33):
                pit = pd.Timedelta(lap * 90, unit="s") if lap == stop_lap else pd.NaT
                rows.append({
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": pd.Timedelta(90 + 0.04 * lap + 0.01 * index, unit="s"),
                    "Compound": ["SOFT", "MEDIUM", "HARD"][index % 3],
                    "TyreLife": lap,
                    "Stint": 1 if lap <= stop_lap else 2,
                    "Position": min(22, index + 1),
                    "GridPosition": min(22, index + 1),
                    "PitInTime": pit,
                    "IsAccurate": True,
                    "TrackStatus": "1",
                })
        return pd.DataFrame(rows)

    def test_build_rows_and_predict(self):
        laps = self.make_laps()
        results = pd.DataFrame({
            "Abbreviation": sorted(FULL_SEASON_DRIVERS),
            "GridPosition": list(range(1, len(FULL_SEASON_DRIVERS) + 1)),
            "TeamName": [DRIVER_TEAM_2026[d] for d in sorted(FULL_SEASON_DRIVERS)],
        })
        rows = build_hazard_rows(laps, results, "Synthetic GP", 32)
        self.assertGreaterEqual(int(rows["StopNextLap"].sum()), 10)
        self.assertTrue((rows["StopNextLap"] == 1).any())
        bundle = fit_first_stop_model(rows)
        forecast = predict_first_stops(bundle, "Synthetic GP", 32, [
            {"driver": "NOR", "team": "McLaren", "grid_position": 2, "starting_compound": "MEDIUM"},
            {"driver": "PIA", "team": "McLaren", "grid_position": 4, "starting_compound": "HARD"},
        ])
        self.assertEqual(forecast["driver_count_forecast"], 2)
        for item in forecast["predictions"]:
            self.assertGreaterEqual(item["probability_stop_before_finish"], 0)
            self.assertLessEqual(item["probability_stop_before_finish"], 1)
            self.assertTrue(item["lap_probabilities"])


if __name__ == "__main__":
    unittest.main()
