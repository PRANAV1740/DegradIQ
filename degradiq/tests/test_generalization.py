"""Regression tests for the offline generalisation-validation harness.

Fast: one small synthetic model fit plus tracked-file and source checks. No network,
no large data. Guards: the harness reports interval WIDTH (not just coverage), the
stop-lap simulation keeps candidate/baseline/interval populations paired, LOCO holds
out the test circuit entirely, and the entry point is offline.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from degradiq.pit_strategy import FEATURE_COLUMNS, LABEL_DEFINITION_VERSION
from degradiq.scripts import validate_generalization as vg

PROV_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiments" / "first_stop_v2"


def _synthetic(n_circuits=6, n_drivers=10, n_laps=26) -> pd.DataFrame:
    rows = []
    for c in range(n_circuits):
        for d in range(n_drivers):
            stop = 8 + ((c + d) % 10)
            for lap in range(1, n_laps + 1):
                rows.append({
                    "LapNumber": float(lap), "RaceLaps": float(n_laps),
                    "LapFraction": lap / n_laps, "TyreLife": float(lap),
                    "GridPosition": float(d + 1), "Position": float(d + 1),
                    "RecentPaceLoss_s": 0.03 * lap,
                    "Compound": ["SOFT", "MEDIUM", "HARD"][d % 3],
                    "Driver": f"D{d}", "Team": f"T{d % 4}", "EventName": f"GP{c}",
                    "StopNextLap": 1 if lap == stop else 0, "Season": 2022 + (c % 3),
                    "LabelDefVersion": LABEL_DEFINITION_VERSION,
                })
    return pd.DataFrame(rows)


class HarnessUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _synthetic()
        train = cls.df[cls.df["EventName"] != "GP0"]
        test = cls.df[cls.df["EventName"] == "GP0"]
        cls.rep = vg.evaluate_split(train, test, "unit")

    def test_evaluate_split_reports_interval_width(self):
        cand = self.rep["candidate"]
        self.assertIn("interval_width_10_90_mean", cand)
        self.assertIn("interval_width_10_90_median", cand)
        self.assertIn("interval_coverage_10_90", cand)
        w = cand["interval_width_10_90_mean"]
        self.assertTrue(w is None or w >= 0)

    def test_evaluate_split_has_probabilistic_and_mae(self):
        cand = self.rep["candidate"]
        for k in ("brier_score", "log_loss", "first_stop_lap_mae"):
            self.assertIn(k, cand)
        self.assertEqual(self.rep["train_circuits"], 5)  # 6 circuits minus the held-out one

    def test_per_event_metrics_present(self):
        self.assertTrue(self.rep["per_event_metrics"])  # at least the held-out GP0


class SimulationPairingTests(unittest.TestCase):
    def test_simulate_populations_are_paired(self):
        # candidate errors, baseline errors, hits and widths must align 1:1.
        df = _synthetic()
        train = df[df["EventName"] != "GP1"]
        test = df[df["EventName"] == "GP1"]
        from degradiq.evaluation import HistoricalMedianBaseline
        from degradiq.scripts.train_first_stop import fit_candidate_first_stop_model
        pipe = fit_candidate_first_stop_model(train, architecture="calibrated_hgb")["pipeline"]
        med = HistoricalMedianBaseline().fit(train)
        sim = vg._simulate_stop_lap(pipe, test, med)
        self.assertEqual(len(sim["_errors"]), sim["drivers_with_observed_first_stop"])
        if sim["interval_width_10_90_mean"] is not None:
            self.assertGreaterEqual(sim["interval_width_10_90_mean"], 0)


class LocoInvariantTests(unittest.TestCase):
    def test_loco_split_excludes_heldout_circuit(self):
        df = _synthetic()
        for c in df["EventName"].unique():
            train = df[df["EventName"].astype(str) != str(c)]
            test = df[df["EventName"].astype(str) == str(c)]
            self.assertNotIn(c, set(train["EventName"]))
            self.assertEqual(set(test["EventName"]), {c})


class TrackedSummaryTests(unittest.TestCase):
    def test_summary_present_with_both_regimes_and_width(self):
        path = PROV_DIR / "generalization_summary.json"
        self.assertTrue(path.exists(), "generalization_summary.json not written")
        s = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("temporal_holdout", s)
        self.assertIn("leave_one_circuit_out", s)
        for r in s["temporal_holdout"]:
            self.assertIn("interval_width_10_90_mean", r["candidate"])
        self.assertIn("interval_width_10_90_mean", s["leave_one_circuit_out"]["macro_mean_over_circuits"])


class EntryPointOfflineTests(unittest.TestCase):
    def test_validate_script_is_offline(self):
        src = (Path(__file__).resolve().parents[1] / "scripts" / "validate_generalization.py").read_text(encoding="utf-8")
        self.assertNotIn("get_session", src)
        self.assertNotIn("get_event_schedule", src)
        self.assertIn("fit_candidate_first_stop_model", src)  # reuses committed training


if __name__ == "__main__":
    unittest.main()
