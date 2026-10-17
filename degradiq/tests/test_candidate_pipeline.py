"""Phase 7 Test Suite: Ingestion rate limiting, checkpoints, zero-leakage training, baselines, and artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from degradiq.rate_limiter import (
    RateLimiter, CheckpointManager, parse_retry_after, SCHEMA_VERSION
)
from degradiq.evaluation import (
    HistoricalMedianBaseline,
    EmpiricalHazardBaseline,
    MarginalHazardBaseline,
    compute_calibration_diagnostics,
)
from degradiq.pit_strategy import (
    FEATURE_COLUMNS, build_hazard_rows, fit_first_stop_model, ARTIFACT_DIR, MODEL_PATH
)
from degradiq.scripts.train_first_stop import (
    evaluate_candidate_chronological, save_candidate_model, fit_candidate_first_stop_model
)


class RateLimiterAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_mgr = CheckpointManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_retry_after(self):
        self.assertEqual(parse_retry_after("12"), 12.0)
        self.assertEqual(parse_retry_after("0"), 0.0)
        self.assertIsNone(parse_retry_after(None))
        self.assertIsNone(parse_retry_after(""))

    def test_rate_pacing(self):
        limiter = RateLimiter(min_interval_s=0.05)
        limiter.last_outbound_time = time.time()
        slept = limiter.pace_outbound_request()
        self.assertGreaterEqual(slept, 0.0)
        self.assertEqual(limiter.total_outbound_requests, 1)

    def test_rate_limiter_retry_limits_on_429(self):
        limiter = RateLimiter(min_interval_s=0.001, max_retries=3, base_backoff_s=0.01, jitter_s=0.005)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "0.01"}

        send_fn = MagicMock(return_value=mock_resp)
        result = limiter.execute_with_retry(send_fn, request_desc="test_429")

        # Must attempt 1 initial + 3 retries = 4 calls total
        self.assertEqual(send_fn.call_count, 4)
        self.assertEqual(result.status_code, 429)
        self.assertEqual(limiter.total_retries, 3)

    def test_checkpoint_atomic_write_and_resume(self):
        df = pd.DataFrame({
            "LapNumber": [1.0, 2.0],
            "RaceLaps": [50.0, 50.0],
            "LapFraction": [0.02, 0.04],
            "TyreLife": [1.0, 2.0],
            "GridPosition": [1.0, 1.0],
            "Position": [1.0, 1.0],
            "RecentPaceLoss_s": [0.0, 0.1],
            "Compound": ["SOFT", "SOFT"],
            "Driver": ["VER", "VER"],
            "Team": ["Red Bull", "Red Bull"],
            "EventName": ["Test GP", "Test GP"],
            "StopNextLap": [0, 1],
        })

        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint(2024, 1, "Test GP", df, status="loaded")
        self.assertTrue(self.checkpoint_mgr.has_valid_checkpoint(2024, 1, FEATURE_COLUMNS, event_name="Test GP"))

        # Resume / load checkpoint
        loaded_df, meta = self.checkpoint_mgr.load_checkpoint(2024, 1, event_name="Test GP")
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(meta["positive_count"], 1)
        self.assertEqual(meta["status"], "loaded")

        # Corrupt file and verify validation rejection
        cpath = self.checkpoint_mgr.checkpoint_path(2024, 1)
        with open(cpath, "wb") as f:
            f.write(b"corrupt non-parquet content")
        self.assertFalse(self.checkpoint_mgr.has_valid_checkpoint(2024, 1, FEATURE_COLUMNS, event_name="Test GP"))

    def test_empty_session_checkpoint(self):
        # Wet race or no eligible rows
        self.checkpoint_mgr.save_checkpoint(2024, 2, "Wet GP", pd.DataFrame(), status="no eligible rows")
        self.assertTrue(self.checkpoint_mgr.has_valid_checkpoint(2024, 2, FEATURE_COLUMNS, event_name="Wet GP"))
        loaded_df, meta = self.checkpoint_mgr.load_checkpoint(2024, 2, event_name="Wet GP")
        self.assertTrue(loaded_df.empty)
        self.assertEqual(meta["status"], "no eligible rows")

    def test_failed_session_not_valid_checkpoint(self):
        # Transient failure must not prevent future retry
        self.checkpoint_mgr.save_checkpoint(2024, 3, "Failed GP", pd.DataFrame(), status="unavailable", error="HTTP 503")
        self.assertFalse(self.checkpoint_mgr.has_valid_checkpoint(2024, 3, FEATURE_COLUMNS, event_name="Failed GP"))


class PipelineAndBaselineTests(unittest.TestCase):
    def make_synthetic_stints(self, season: int, event_name: str) -> pd.DataFrame:
        drivers = [
            "VER", "NOR", "HAM", "LEC", "PIA", "RUS", "SAI", "PER",
            "ALO", "STR", "TSU", "OCO"
        ]
        rows = []
        for d_idx, driver in enumerate(drivers):
            stop_lap = 10 + (d_idx % 8)
            compound = ["SOFT", "MEDIUM", "HARD"][d_idx % 3]
            for lap in range(1, 30):
                is_stop = 1 if lap == stop_lap else 0
                rows.append({
                    "LapNumber": float(lap),
                    "RaceLaps": 50.0,
                    "LapFraction": float(lap / 50.0),
                    "TyreLife": float(lap),
                    "GridPosition": float(d_idx + 1),
                    "Position": float(d_idx + 1),
                    "RecentPaceLoss_s": 0.05 * (lap - 1),
                    "Compound": compound,
                    "Driver": driver,
                    "Team": f"Team_{driver}",
                    "EventName": event_name,
                    "StopNextLap": is_stop,
                    "Season": season,
                })
        return pd.DataFrame(rows)

    def test_drive_through_penalty_not_labeled_as_stop(self):
        """Verify pit stop without stint or compound change is excluded from first-stop label."""
        laps = pd.DataFrame([
            {"Driver": "VER", "LapNumber": 10, "LapTime": pd.Timedelta(90, unit="s"), "Compound": "MEDIUM", "TyreLife": 10, "Stint": 1, "PitInTime": pd.Timedelta(900, unit="s"), "Position": 1, "IsAccurate": True, "TrackStatus": "1"},
            {"Driver": "VER", "LapNumber": 11, "LapTime": pd.Timedelta(115, unit="s"), "Compound": "MEDIUM", "TyreLife": 11, "Stint": 1, "PitInTime": pd.NaT, "Position": 5, "IsAccurate": True, "TrackStatus": "1"},
            {"Driver": "VER", "LapNumber": 12, "LapTime": pd.Timedelta(90, unit="s"), "Compound": "MEDIUM", "TyreLife": 12, "Stint": 1, "PitInTime": pd.NaT, "Position": 5, "IsAccurate": True, "TrackStatus": "1"},
        ])
        hazard_df = build_hazard_rows(laps, race_laps=50)
        # Drive-through on lap 10 with same stint 1 and same compound must not be labelled as pit stop
        self.assertEqual(int(hazard_df["StopNextLap"].sum()), 0)

    def test_retirement_treated_as_right_censored(self):
        """Verify driver retiring without pit stop is right-censored with zero positive stops."""
        laps = pd.DataFrame([
            {"Driver": "ALB", "LapNumber": lap, "LapTime": pd.Timedelta(92, unit="s"), "Compound": "HARD", "TyreLife": lap, "Stint": 1, "PitInTime": pd.NaT, "Position": 12, "IsAccurate": True, "TrackStatus": "1"}
            for lap in range(1, 15)
        ])
        hazard_df = build_hazard_rows(laps, race_laps=50)
        # 14 laps total -> 13 hazard transition rows
        self.assertEqual(len(hazard_df), 13)
        # No pit stops occurred: all rows are right-censored non-stops
        self.assertEqual(int(hazard_df["StopNextLap"].sum()), 0)
        # No post-retirement laps exist
        self.assertEqual(hazard_df["LapNumber"].max(), 13.0)

    def test_pre_race_features_contain_no_post_start_information(self):
        """Verify pre-race feature generation does not leak post-start pace or position telemetry."""
        from degradiq.pit_strategy import _lap_features
        driver_state = {
            "driver": "NOR",
            "grid_position": 3,
            "starting_compound": "MEDIUM",
            "starting_tyre_age": 1,
            "current_position": 1,  # Post-start information
            "recent_pace_loss_s": 0.45,  # Post-start telemetry
        }
        feat = _lap_features(driver_state, lap_number=15, race_laps=57, event_name="Bahrain Grand Prix", mode="pre-race")
        # Pre-race must ignore current_position and recent_pace_loss_s
        self.assertEqual(feat["Position"], 3.0)
        self.assertTrue(np.isnan(feat["RecentPaceLoss_s"]))
        self.assertEqual(feat["Compound"], "MEDIUM")

    def test_live_features_strictly_use_current_lap_information(self):
        """Verify live feature generation uses current lap state without future knowledge."""
        from degradiq.pit_strategy import _lap_features
        driver_state = {
            "driver": "HAM",
            "grid_position": 8,
            "current_lap": 12,
            "current_position": 5,
            "current_compound": "HARD",
            "current_tyre_age": 12,
            "recent_pace_loss_s": 0.18,
        }
        feat = _lap_features(driver_state, lap_number=13, race_laps=57, event_name="Bahrain Grand Prix", mode="live")
        self.assertEqual(feat["Position"], 5.0)
        self.assertEqual(feat["RecentPaceLoss_s"], 0.18)
        self.assertEqual(feat["Compound"], "HARD")
        self.assertEqual(feat["TyreLife"], 12.0)

    def test_zero_leakage_and_baseline_evaluation(self):
        train_df1 = self.make_synthetic_stints(2022, "Bahrain GP")
        train_df2 = self.make_synthetic_stints(2023, "Saudi GP")
        holdout_df = self.make_synthetic_stints(2024, "Australia GP")
        combined = pd.concat([train_df1, train_df2, holdout_df], ignore_index=True)

        coverage = pd.DataFrame([
            {"year": 2022, "event": "Bahrain GP", "status": "loaded"},
            {"year": 2023, "event": "Saudi GP", "status": "loaded"},
            {"year": 2024, "event": "Australia GP", "status": "loaded"},
            {"year": 2024, "event": "China GP", "status": "unavailable"},
        ])

        bundle, report = evaluate_candidate_chronological(combined, coverage, holdout_year=2024)

        # Incomplete because 1 of 2 scheduled 2024 events is unavailable
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("China GP", report["missing_events"])
        self.assertEqual(report["train_rows"], len(train_df1) + len(train_df2))
        self.assertEqual(report["holdout_rows"], len(holdout_df))

        # Check all 3 baselines were evaluated
        metrics = report["metrics"]
        self.assertIn("candidate_model", metrics)
        self.assertIn("baseline_1_historical_median", metrics)
        self.assertIn("baseline_2_empirical_hazard", metrics)
        self.assertIn("baseline_3_marginal_hazard", metrics)

        # Baseline MAE and Brier score must be non-None
        self.assertIsNotNone(metrics["candidate_model"]["first_stop_lap_mae"])
        self.assertIsNotNone(metrics["baseline_1_historical_median"]["first_stop_lap_mae"])
        self.assertIsNotNone(metrics["candidate_model"]["brier_score"])
        self.assertIsNotNone(metrics["baseline_2_empirical_hazard"]["brier_score"])

    def test_fit_candidate_derives_training_seasons_from_data(self):
        """Candidate bundle records the seasons actually fitted, not a hardcoded default."""
        combined = pd.concat([
            self.make_synthetic_stints(2022, "Bahrain GP"),
            self.make_synthetic_stints(2023, "Saudi GP"),
            self.make_synthetic_stints(2024, "Australia GP"),
        ], ignore_index=True)
        bundle = fit_candidate_first_stop_model(combined, architecture="calibrated_hgb")
        self.assertEqual(bundle["training_seasons"], [2022, 2023, 2024])

    def test_save_candidate_model_records_correct_training_seasons(self):
        """Explicit season list and bundle-derived seasons both override the stale default."""
        train_df = self.make_synthetic_stints(2022, "Bahrain GP")

        # (a) Explicit parameter wins.
        bundle = fit_first_stop_model(train_df)  # no training_seasons key
        save_candidate_model(bundle, {"test_note": "season_param"}, version="season_explicit_tmp",
                             training_seasons=[2022, 2023, 2024])
        meta_a = ARTIFACT_DIR / "candidate_first_stop_model_season_explicit_tmp_metadata.json"
        model_a = ARTIFACT_DIR / "candidate_first_stop_model_season_explicit_tmp.joblib"

        # (b) Bundle-derived seasons used when no explicit parameter is given.
        bundle_b = dict(bundle)
        bundle_b["training_seasons"] = [2022, 2023]
        save_candidate_model(bundle_b, {"test_note": "season_bundle"}, version="season_bundle_tmp")
        meta_b = ARTIFACT_DIR / "candidate_first_stop_model_season_bundle_tmp_metadata.json"
        model_b = ARTIFACT_DIR / "candidate_first_stop_model_season_bundle_tmp.joblib"

        try:
            self.assertEqual(json.loads(meta_a.read_text())["training_seasons"], [2022, 2023, 2024])
            self.assertEqual(json.loads(meta_b.read_text())["training_seasons"], [2022, 2023])
        finally:
            for p in (meta_a, model_a, meta_b, model_b):
                if p.exists():
                    p.unlink()

    def test_baseline_and_candidate_evaluated_on_identical_drivers(self):
        """Reported baseline and candidate MAE populations must contain the same drivers."""
        combined = pd.concat([
            self.make_synthetic_stints(2022, "Bahrain GP"),
            self.make_synthetic_stints(2023, "Saudi GP"),
            self.make_synthetic_stints(2024, "Australia GP"),
        ], ignore_index=True)
        coverage = pd.DataFrame([
            {"year": 2022, "event": "Bahrain GP", "status": "loaded"},
            {"year": 2023, "event": "Saudi GP", "status": "loaded"},
            {"year": 2024, "event": "Australia GP", "status": "loaded"},
        ])
        _, report = evaluate_candidate_chronological(combined, coverage, holdout_year=2024)
        self.assertEqual(
            report["metrics"]["baseline_1_historical_median"]["drivers_evaluated"],
            report["drivers_with_observed_first_stop"],
        )

    def test_candidate_artifacts_do_not_overwrite_production(self):
        prod_model_path = ARTIFACT_DIR / "first_stop_model.joblib"
        prod_meta_path = ARTIFACT_DIR / "first_stop_model_metadata.json"
        orig_prod_model_bytes = prod_model_path.read_bytes() if prod_model_path.exists() else None
        orig_prod_meta_bytes = prod_meta_path.read_bytes() if prod_meta_path.exists() else None

        train_df = self.make_synthetic_stints(2022, "Bahrain GP")
        bundle = fit_first_stop_model(train_df)
        save_candidate_model(bundle, {"test_note": "unit_test_run"}, version="test_tmp")

        cand_model_path = ARTIFACT_DIR / "candidate_first_stop_model_test_tmp.joblib"
        cand_meta_path = ARTIFACT_DIR / "candidate_first_stop_model_test_tmp_metadata.json"

        try:
            self.assertTrue(cand_model_path.exists())
            self.assertTrue(cand_meta_path.exists())

            # Production model MUST NOT be altered
            if orig_prod_model_bytes is not None:
                self.assertEqual(prod_model_path.read_bytes(), orig_prod_model_bytes)
            if orig_prod_meta_bytes is not None:
                self.assertEqual(prod_meta_path.read_bytes(), orig_prod_meta_bytes)
        finally:
            if cand_model_path.exists():
                cand_model_path.unlink()
            if cand_meta_path.exists():
                cand_meta_path.unlink()


class StreamlitDashboardTest(unittest.TestCase):
    def test_dashboard_renders_without_exceptions(self):
        at = AppTest.from_file(r"p:\DegradIQ\degradiq\dashboard.py", default_timeout=20)
        at.run()
        self.assertFalse(at.exception, f"Dashboard threw unexpected exception: {at.exception}")
        self.assertTrue(len(at.title) > 0)
        self.assertEqual(at.title[0].value, "DegradIQ")


if __name__ == "__main__":
    unittest.main()
