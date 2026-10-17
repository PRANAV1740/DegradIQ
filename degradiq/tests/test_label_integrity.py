"""Phase 3 regression tests for first-stop label integrity.

Covers: missing stop-lap LapTime detection, risk-set truncation, VSC/SC-eligible
observations, confirmed red-flag exclusion, unresolved timing, drive-through
exclusion, absent predicting row (no relocation), future-information isolation,
and target indexing. All fixtures are synthetic; no real F1 result is implied.
"""
from __future__ import annotations
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from degradiq.pit_strategy import build_hazard_rows, FEATURE_COLUMNS, LABEL_DEFINITION_VERSION
from degradiq.rate_limiter import CheckpointManager, RateLimiter
from degradiq.scripts.train_first_stop import (
    label_compatibility_report, _incompatible_contributing_events,
    assert_training_label_compatibility, fit_candidate_first_stop_model,
    _fit_calibrated_hazard_model, collect_rows_with_checkpoints, _race_timing_cached,
)


def _trainable_frame(version=None, n_drivers=15, n_laps=30):
    """A directly-built, fittable candidate training frame (FEATURE_COLUMNS + StopNextLap)."""
    rows = []
    for d in range(n_drivers):
        stop = 10 + (d % 6)
        for lap in range(1, n_laps + 1):
            rows.append({
                "LapNumber": float(lap), "RaceLaps": float(n_laps), "LapFraction": lap / n_laps,
                "TyreLife": float(lap), "GridPosition": float(d + 1), "Position": float(d + 1),
                "RecentPaceLoss_s": 0.05 * lap, "Compound": ["SOFT", "MEDIUM", "HARD"][d % 3],
                "Driver": f"D{d}", "Team": f"T{d}", "EventName": "GP",
                "StopNextLap": 1 if lap == stop else 0, "Season": 2022,
            })
    df = pd.DataFrame(rows)
    if version is not None:
        df["LabelDefVersion"] = version
    return df


def gen_race(stop_lap, n=40, start_c="HARD", stop_c="HARD",
             nat_at=(), nat_time_at=(), ts_overrides=None, acc_false=(),
             with_time=False, pit=True):
    """One-driver synthetic race. Stint 2 starts the lap AFTER stop_lap with reset age."""
    ts_overrides = ts_overrides or {}
    rows = []
    for lap in range(1, n + 1):
        in_s2 = stop_lap is not None and lap > stop_lap
        stint = 2 if in_s2 else 1
        comp = stop_c if in_s2 else start_c
        age = (lap - stop_lap) if in_s2 else lap               # reset to 1 on out-lap
        lt = pd.NaT if lap in nat_at else pd.Timedelta(90.0 + 0.05 * lap, unit="s")
        pit_s = (lap * 90 - 20) if (pit and stop_lap is not None and lap == stop_lap) else None
        time_s = None if (lap in nat_time_at or not with_time) else lap * 90
        rows.append({
            "Driver": "VER", "LapNumber": lap, "LapTime": lt, "Compound": comp,
            "TyreLife": age, "Stint": stint,
            "PitInTime": pd.Timedelta(pit_s, unit="s") if pit_s is not None else pd.NaT,
            "PitOutTime": pd.NaT, "Position": 5, "GridPosition": 5,
            "IsAccurate": lap not in acc_false,
            "TrackStatus": ts_overrides.get(lap, "1"),
            "Time": pd.Timedelta(time_s, unit="s") if time_s is not None else pd.NaT,
        })
    return pd.DataFrame(rows)


def ts_frame(changes):
    return pd.DataFrame({
        "Time": [pd.Timedelta(t, unit="s") for t, _ in changes],
        "Status": [s for _, s in changes],
        "Message": [""] * len(changes),
    })


class LabelIntegrityTests(unittest.TestCase):

    def test_1_missing_stop_lap_laptime_detected(self):
        """Austria/Magnussen: pit lap with NaT LapTime is still detected; lap k-1 positive; no post-stop rows."""
        laps = gen_race(stop_lap=34, n=40, start_c="HARD", stop_c="HARD", nat_at=(34,))
        hz = build_hazard_rows(laps, None, "T", 40)
        self.assertEqual(int(hz["StopNextLap"].sum()), 1)
        pos = hz[hz["StopNextLap"] == 1]
        self.assertEqual(int(pos["LapNumber"].iloc[0]), 33)
        self.assertTrue((hz["LapNumber"] < 34).all())           # no row at/after the stop
        self.assertNotIn(34.0, set(hz["LapNumber"]))

    def test_2_normal_green_stop(self):
        laps = gen_race(stop_lap=20, stop_c="MEDIUM", start_c="HARD")
        hz = build_hazard_rows(laps, None, "T", 40)
        self.assertEqual(int(hz["StopNextLap"].sum()), 1)
        self.assertEqual(int(hz[hz["StopNextLap"] == 1]["LapNumber"].iloc[0]), 19)
        self.assertTrue((hz["LapNumber"] < 20).all())

    def test_3_vsc_strategic_stop_positive_on_preceding_lap(self):
        """Stop under VSC: the immediately preceding caution lap gets the positive, not an earlier lap."""
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True, ts_overrides={14: "6", 15: "6"})
        ts = ts_frame([(0, "1"), (13 * 90 + 45, "6"), (15 * 90 + 45, "1")])
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        pos = hz[hz["StopNextLap"] == 1]
        self.assertEqual(int(pos["StopNextLap"].sum()), 1)
        self.assertEqual(int(pos["LapNumber"].iloc[0]), 14)     # exactly one lap before the stop
        self.assertFalse((hz[hz["LapNumber"] < 14]["StopNextLap"] == 1).any())

    def test_4_confirmed_red_flag_entry_excluded(self):
        """Pit entry confirmed (by timestamp) during an active red flag is not a strategic stop."""
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        ts = ts_frame([(0, "1"), (14 * 90, "5"), (16 * 90, "1")])  # red covers the pit entry
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)

    def test_5_ambiguous_timing_unresolved_not_guessed(self):
        """Predicting-lap state cannot be established -> unresolved, no guessed positive."""
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True, nat_time_at=(14,))
        ts = ts_frame([(0, "1")])
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)
        reasons = [u["reason"] for u in hz.attrs.get("unresolved_events", [])]
        self.assertIn("observation_state_unresolved", reasons)

    def test_6_drive_through_not_labelled(self):
        laps = pd.DataFrame([
            {"Driver": "VER", "LapNumber": lap, "LapTime": pd.Timedelta(90, unit="s"),
             "Compound": "HARD", "TyreLife": lap, "Stint": 1,
             "PitInTime": pd.Timedelta(900, unit="s") if lap == 10 else pd.NaT,
             "Position": 5, "IsAccurate": True, "TrackStatus": "1"}
            for lap in range(1, 21)
        ])
        hz = build_hazard_rows(laps, None, "T", 40)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)

    def test_7_absent_predicting_row_no_relocation(self):
        """If lap k-1 is missing, do not relocate the positive; mark unresolved; emit no post-stop rows."""
        laps = gen_race(stop_lap=20, n=25, stop_c="MEDIUM")
        laps = laps[laps["LapNumber"] != 19].copy()             # remove the predicting row
        hz = build_hazard_rows(laps, None, "T", 40)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)       # no backward relocation
        reasons = [u["reason"] for u in hz.attrs.get("unresolved_events", [])]
        self.assertIn("predicting_row_absent", reasons)
        self.assertTrue((hz["LapNumber"] < 20).all())

    def test_8_risk_set_invariant_after_stop(self):
        for stop in (34, 20, 10):
            laps = gen_race(stop_lap=stop, n=max(stop + 6, 40), stop_c="MEDIUM")
            hz = build_hazard_rows(laps, None, "T", max(stop + 6, 40))
            self.assertTrue((hz["LapNumber"] < stop).all(), f"row at/after stop {stop}")

    def test_9_future_information_isolation(self):
        base = gen_race(stop_lap=20, stop_c="MEDIUM")
        altered = base.copy()
        altered.loc[altered["LapNumber"] == 9, "LapTime"] = pd.Timedelta(300, unit="s")  # future lap
        hz_b = build_hazard_rows(base, None, "T", 40)
        hz_a = build_hazard_rows(altered, None, "T", 40)
        row_b = hz_b[hz_b["LapNumber"] == 8].iloc[0]
        row_a = hz_a[hz_a["LapNumber"] == 8].iloc[0]
        for col in ("RecentPaceLoss_s", "TyreLife", "StopNextLap", "LapFraction"):
            self.assertTrue((pd.isna(row_b[col]) and pd.isna(row_a[col])) or row_b[col] == row_a[col],
                            f"lap-8 feature {col} changed when a FUTURE lap changed")

    def test_10_target_indexing_matches_actual_pit_lap(self):
        for stop in (34, 20, 15):
            laps = gen_race(stop_lap=stop, n=40, stop_c="MEDIUM", with_time=True,
                            ts_overrides={})
            ts = ts_frame([(0, "1")])
            hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
            pos = hz[hz["StopNextLap"] == 1]
            self.assertEqual(len(pos), 1)
            self.assertEqual(int(pos["LapNumber"].iloc[0]) + 1, stop)
            # the stop lap actually carried a PitInTime in the raw input
            self.assertTrue(laps[laps["LapNumber"] == stop]["PitInTime"].notna().iloc[0])

    def test_extra_column_not_in_feature_schema(self):
        """LapTimeMissing is emitted but must not be part of FEATURE_COLUMNS (model schema preserved)."""
        laps = gen_race(stop_lap=20, stop_c="MEDIUM", nat_at=(5,))
        hz = build_hazard_rows(laps, None, "T", 40)
        self.assertIn("LapTimeMissing", hz.columns)
        self.assertNotIn("LapTimeMissing", FEATURE_COLUMNS)
        self.assertEqual(int(hz[hz["LapNumber"] == 5]["LapTimeMissing"].iloc[0]), 1)


class ContractAndDiagnosticsTests(unittest.TestCase):
    """Phase 3B: eligible-state contract, diagnostics propagation, fail-safe classification,
    timestamp edge cases, and the checkpoint label-version gate."""

    # --- Blocker 3: frozen eligible-state contract (yellow excluded) ---
    def test_yellow_only_prediction_row_not_eligible(self):
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True, ts_overrides={14: "2", 15: "2"})
        ts = ts_frame([(0, "1"), (13 * 90 + 45, "2"), (15 * 90 + 45, "1")])  # yellow spans lap 14
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)
        reasons = [u["reason"] for u in hz.attrs.get("unresolved_events", [])]
        self.assertIn("predicting_lap_state_2", reasons)

    # --- Blocker 2: diagnostics survive multi-event aggregation ---
    def test_diagnostics_survive_aggregation(self):
        sink: list = []
        e1 = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)   # resolved
        ts1 = ts_frame([(0, "1")])
        f1 = build_hazard_rows(e1, None, "E1", 40, track_status=ts1, diagnostics_sink=sink)
        e2 = gen_race(stop_lap=20, stop_c="MEDIUM", pit=False, with_time=True)  # NaT pit-time -> unresolved
        f2 = build_hazard_rows(e2, None, "E2", 40, track_status=ts_frame([(0, "1")]), diagnostics_sink=sink)
        combined = pd.concat([f1, f2], ignore_index=True)
        self.assertEqual(combined.attrs.get("unresolved_events", "GONE"), "GONE")  # attrs dropped by concat
        self.assertEqual([d["event"] for d in sink], ["E1", "E2"])                # sink intact
        self.assertEqual(sink[0]["unresolved_count"], 0)
        self.assertGreaterEqual(sink[1]["unresolved_count"], 1)

    def test_zero_unresolved_distinct_from_unknown(self):
        sink: list = []
        clean = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        build_hazard_rows(clean, None, "C", 40, track_status=ts_frame([(0, "1")]), diagnostics_sink=sink)
        self.assertEqual(sink[0]["unresolved_count"], 0)              # a real, known zero
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            df = gen_race(stop_lap=10, stop_c="MEDIUM").assign(Season=2022)
            mgr.save_checkpoint(2099, 1, "Legacy GP", df, status="loaded")  # no version -> legacy
            self.assertIsNone(mgr.checkpoint_label_version(2099, 1, event_name="Legacy GP"))  # unknown, NOT 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_legacy_checkpoint_not_certified_clean(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            df = build_hazard_rows(gen_race(stop_lap=10, stop_c="MEDIUM"), None, "X", 40)  # valid schema
            mgr.save_checkpoint(2099, 1, "Legacy GP", df, status="loaded")                       # legacy
            mgr.save_checkpoint(2099, 2, "Fresh GP", df, status="loaded",
                                label_definition_version=LABEL_DEFINITION_VERSION,
                                diagnostics={"unresolved_count": 0, "unresolved_events": [],
                                             "timeline_available": True})                         # compatible
            rep = label_compatibility_report(mgr, [(2099, 1, "Legacy GP"), (2099, 2, "Fresh GP")])
            by_event = dict(zip(rep["event"], rep["compatibility"]))
            self.assertEqual(by_event["Legacy GP"], "legacy_unknown")
            self.assertEqual(by_event["Fresh GP"], "compatible")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- Blocker 4: fail-safe unclassifiable pit entry ---
    def test_pit_entry_before_timeline_is_unresolved(self):
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        ts = ts_frame([(16 * 90, "1")])          # timeline starts AFTER the pit entry -> unconfirmable
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)
        self.assertIn("pit_entry_state_unconfirmed", [u["reason"] for u in hz.attrs["unresolved_events"]])
        self.assertTrue((hz["LapNumber"] < 15).all())   # risk set truncated even though unresolved

    def test_tyre_change_without_pit_timestamp_is_unresolved(self):
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", pit=False, with_time=True)  # stint change, no PitInTime
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts_frame([(0, "1")]))
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)
        self.assertIn("pit_entry_timestamp_missing", [u["reason"] for u in hz.attrs["unresolved_events"]])
        self.assertTrue((hz["LapNumber"] < 15).all())

    def test_nat_timeline_not_treated_as_no_red_flag(self):
        # Pit entry exactly at a confirmed red-flag deploy boundary -> excluded (inclusive boundary).
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        entry = 15 * 90 - 20
        ts = ts_frame([(0, "1"), (entry, "5"), (2000, "1")])
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        self.assertEqual(int(hz["StopNextLap"].sum()), 0)

    def test_observation_cutoff_has_no_lookahead(self):
        # A transition to an INELIGIBLE state that occurs AFTER lap k's completion
        # must not retroactively make lap k ineligible: eligibility uses end-of-lap-k state.
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        # lap 14 completes at 14*90=1260s (green); a yellow is deployed at 1290s (during lap 15).
        ts = ts_frame([(0, "1"), (14 * 90 + 30, "2"), (16 * 90, "1")])
        hz = build_hazard_rows(laps, None, "T", 40, track_status=ts)
        pos = hz[hz["StopNextLap"] == 1]
        self.assertEqual(len(pos), 1)
        self.assertEqual(int(pos["LapNumber"].iloc[0]), 14)   # green at end of lap 14 -> eligible

    # --- Blocker 5: timestamp edge cases ---
    def test_out_of_order_timestamps_normalized(self):
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True, ts_overrides={14: "6", 15: "6"})
        unordered = ts_frame([(15 * 90 + 45, "1"), (0, "1"), (13 * 90 + 45, "6")])
        ordered = ts_frame([(0, "1"), (13 * 90 + 45, "6"), (15 * 90 + 45, "1")])
        a = build_hazard_rows(laps, None, "T", 40, track_status=unordered)
        b = build_hazard_rows(laps, None, "T", 40, track_status=ordered)
        self.assertEqual(int(a["StopNextLap"].sum()), int(b["StopNextLap"].sum()))
        self.assertEqual(int(a[a["StopNextLap"] == 1]["LapNumber"].iloc[0]), 14)

    def test_duplicate_timestamps_deterministic(self):
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", with_time=True)
        dup = ts_frame([(0, "1"), (14 * 90, "6"), (14 * 90, "1")])  # same Time, two statuses
        r1 = build_hazard_rows(laps, None, "T", 40, track_status=dup)
        r2 = build_hazard_rows(laps, None, "T", 40, track_status=dup)
        self.assertEqual(int(r1["StopNextLap"].sum()), int(r2["StopNextLap"].sum()))  # stable

    def test_predicting_row_invalid_laptime_still_labelled(self):
        # Predicting lap has NaT LapTime but is a valid (green, accurate) observation.
        laps = gen_race(stop_lap=15, stop_c="MEDIUM", nat_at=(14,))
        hz = build_hazard_rows(laps, None, "T", 40)
        pos = hz[hz["StopNextLap"] == 1]
        self.assertEqual(len(pos), 1)
        self.assertEqual(int(pos["LapNumber"].iloc[0]), 14)
        self.assertEqual(int(pos["LapTimeMissing"].iloc[0]), 1)
        # Distinct from IsAccurate=False: an inaccurate predicting lap must NOT be labelled.
        laps2 = gen_race(stop_lap=15, stop_c="MEDIUM", acc_false=(14,))
        hz2 = build_hazard_rows(laps2, None, "T", 40)
        self.assertEqual(int(hz2["StopNextLap"].sum()), 0)

    def test_missing_track_status_documented_fallback(self):
        sink: list = []
        laps = gen_race(stop_lap=15, stop_c="MEDIUM")
        hz = build_hazard_rows(laps, None, "T", 40, track_status=None, diagnostics_sink=sink)
        self.assertIs(sink[0]["timeline_available"], False)      # explicit, not silent
        self.assertEqual(int(hz["StopNextLap"].sum()), 1)        # green-only fallback still labels green stop

    # --- Blocker 6: label-definition version gate ---
    def test_version_gate_blocks_legacy_and_mixed(self):
        compatible = pd.DataFrame([
            {"year": 2022, "event": "A", "status": "loaded", "label_definition_version": LABEL_DEFINITION_VERSION},
            {"year": 2023, "event": "B", "status": "loaded", "label_definition_version": LABEL_DEFINITION_VERSION},
        ])
        self.assertEqual(_incompatible_contributing_events(compatible, LABEL_DEFINITION_VERSION), [])
        mixed = pd.DataFrame([
            {"year": 2022, "event": "A", "status": "loaded", "label_definition_version": LABEL_DEFINITION_VERSION},
            {"year": 2023, "event": "B", "status": "loaded", "label_definition_version": None},       # legacy
            {"year": 2024, "event": "C", "status": "loaded", "label_definition_version": "1.0-old"},  # stale
        ])
        offenders = _incompatible_contributing_events(mixed, LABEL_DEFINITION_VERSION)
        self.assertEqual({o[1] for o in offenders}, {"B", "C"})

    # --- Blocker 5.9: all production call sites pass track_status ---
    def test_production_call_sites_pass_track_status(self):
        root = Path(__file__).resolve().parents[1]
        for rel in ("scripts/train_first_stop.py", "scripts/recover_2025.py"):
            src = (root / rel).read_text(encoding="utf-8")
            self.assertIn("build_hazard_rows(", src)
            self.assertIn("track_status=getattr(session", src, f"{rel} must pass track_status")


class VersionGateAndDiagnosticsPersistenceTests(unittest.TestCase):
    """Phase 3C: fail-closed version gate and checkpoint diagnostic persistence."""

    # --- Blocker 1: fail-closed gate ---
    def test_fit_rejects_stale_version_before_fitting(self):
        df = _trainable_frame(version="1.0-old")
        with self.assertRaises(RuntimeError) as cm:
            fit_candidate_first_stop_model(df, architecture="calibrated_hgb")
        self.assertIn("Fail-closed", str(cm.exception))   # gate fired, not the fitter

    def test_fit_rejects_mixed_versions(self):
        df = _trainable_frame(version=LABEL_DEFINITION_VERSION)
        df.loc[df["Driver"] == "D0", "LabelDefVersion"] = "1.0-old"
        with self.assertRaises(RuntimeError):
            fit_candidate_first_stop_model(df, architecture="calibrated_hgb")

    def test_fit_rejects_unversioned_when_column_has_nulls(self):
        df = _trainable_frame(version=LABEL_DEFINITION_VERSION)
        df.loc[df["Driver"] == "D1", "LabelDefVersion"] = None
        with self.assertRaises(RuntimeError):
            fit_candidate_first_stop_model(df, architecture="calibrated_hgb")

    def test_fit_accepts_current_version(self):
        bundle = fit_candidate_first_stop_model(_trainable_frame(version=LABEL_DEFINITION_VERSION),
                                                architecture="calibrated_hgb")
        self.assertIn("pipeline", bundle)

    def test_fit_rejects_absent_certification(self):
        # Fail-closed: the STANDARD API must reject uncertified data (no bypass).
        with self.assertRaises(RuntimeError) as cm:
            fit_candidate_first_stop_model(_trainable_frame(version=None), architecture="calibrated_hgb")
        self.assertIn("Fail-closed", str(cm.exception))

    def test_low_level_fitter_allows_uncertified_synthetic(self):
        # The separated low-level fitter is the ONLY uncertified path (tests/internal only).
        bundle = _fit_calibrated_hazard_model(_trainable_frame(version=None), architecture="calibrated_hgb")
        self.assertIn("pipeline", bundle)

    def test_collect_rows_defaults_to_train_mode_and_validates_mode(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            with self.assertRaises(ValueError):
                collect_rows_with_checkpoints(2022, 2022, mgr, RateLimiter(), mode="bogus")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_gate_helper_flags_legacy_stale_mixed(self):
        status = pd.DataFrame([
            {"year": 2022, "event": "A", "status": "loaded", "train_compatible": True,
             "label_definition_version": LABEL_DEFINITION_VERSION},
            {"year": 2022, "event": "B", "status": "loaded", "train_compatible": False,
             "label_definition_version": None},               # legacy
            {"year": 2022, "event": "C", "status": "loaded", "train_compatible": False,
             "label_definition_version": "1.0-old"},           # stale
            {"year": 2022, "event": "D", "status": "unavailable", "train_compatible": False,
             "label_definition_version": None},                # not row-contributing
        ])
        offenders = {o[1] for o in _incompatible_contributing_events(status)}
        self.assertEqual(offenders, {"B", "C"})               # D excluded (not loaded)
        with self.assertRaises(RuntimeError):
            assert_training_label_compatibility(status)

    def test_training_entrypoints_are_fail_closed(self):
        root = Path(__file__).resolve().parents[1]
        tfs = (root / "scripts/train_first_stop.py").read_text(encoding="utf-8")
        self.assertIn('mode: str = "train"', tfs)             # default is fail-closed
        self.assertIn("LabelDefVersion", tfs)                 # fit-level defense-in-depth
        ev = (root / "scripts/evaluate_2025_holdout.py").read_text(encoding="utf-8")
        self.assertIn("collect_rows_with_checkpoints", ev)
        self.assertNotIn('mode="inspect"', ev)                # must not bypass the gate

    # --- Blocker 3: diagnostics persist through checkpoint write/read ---
    def test_diagnostics_persist_through_roundtrip(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            df = build_hazard_rows(gen_race(stop_lap=10, stop_c="MEDIUM"), None, "X", 40)
            diag = {"unresolved_count": 2, "timeline_available": True,
                    "unresolved_events": [{"reason": "pit_entry_timestamp_missing"},
                                          {"reason": "predicting_lap_state_2"}]}
            mgr.save_checkpoint(2099, 5, "RT GP", df, status="loaded",
                                label_definition_version=LABEL_DEFINITION_VERSION, diagnostics=diag)
            back = mgr.checkpoint_diagnostics(2099, 5, event_name="RT GP")
            self.assertIsNotNone(back)
            self.assertEqual(back["unresolved_count"], 2)
            self.assertEqual(back["unresolved_reasons"]["pit_entry_timestamp_missing"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_diagnostics_is_none_not_zero(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            df = build_hazard_rows(gen_race(stop_lap=10, stop_c="MEDIUM"), None, "X", 40)
            mgr.save_checkpoint(2099, 6, "Legacy GP", df, status="loaded")  # no diagnostics
            self.assertIsNone(mgr.checkpoint_diagnostics(2099, 6, event_name="Legacy GP"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_version_present_but_diagnostics_missing_is_legacy_unknown(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = CheckpointManager(tmp)
            df = build_hazard_rows(gen_race(stop_lap=10, stop_c="MEDIUM"), None, "X", 40)
            mgr.save_checkpoint(2099, 7, "Versioned No Diag", df, status="loaded",
                                label_definition_version=LABEL_DEFINITION_VERSION)  # version, no diagnostics
            rep = label_compatibility_report(mgr, [(2099, 7, "Versioned No Diag")])
            self.assertEqual(rep["compatibility"].iloc[0], "legacy_unknown")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CacheMissClassificationTests(unittest.TestCase):
    """Phase 4-5 audit: a genuine cache miss (timing payload absent) must classify as
    cache_miss, not processing_failure; locally-present data that fails classifies as
    processing_failure."""

    def _event_dir(self, root, year, date, name):
        d = Path(root) / str(year) / f"{date}_{name.replace(' ', '_')}" / f"{date}_Race"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_absent_timing_is_cache_miss(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            # Event dir exists with only a lightweight session_info file (no timing payload).
            d = self._event_dir(tmp, 2025, "2025-07-06", "British Grand Prix")
            (d / "session_info.ff1pkl").write_bytes(b"x")
            self.assertFalse(_race_timing_cached(tmp, 2025, pd.Timestamp("2025-07-06"), "British Grand Prix"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_event_dir_is_cache_miss(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (Path(tmp) / "2025").mkdir(parents=True, exist_ok=True)
            self.assertFalse(_race_timing_cached(tmp, 2025, pd.Timestamp("2025-07-06"), "British Grand Prix"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_present_timing_is_not_cache_miss(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = self._event_dir(tmp, 2025, "2025-03-16", "Australian Grand Prix")
            (d / "_extended_timing_data.ff1pkl").write_bytes(b"x")  # real timing payload present
            self.assertTrue(_race_timing_cached(tmp, 2025, pd.Timestamp("2025-03-16"), "Australian Grand Prix"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_present_timing_without_eventdate_fallback(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = self._event_dir(tmp, 2024, "2024-09-15", "Azerbaijan Grand Prix")
            (d / "timing_app_data.ff1pkl").write_bytes(b"x")
            # event_date None -> name-suffix fallback still finds it
            self.assertTrue(_race_timing_cached(tmp, 2024, None, "Azerbaijan Grand Prix"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_schedule_unavailable_status_path_exists(self):
        # The schedule-unavailable status is still emitted explicitly (not reclassified).
        src = (Path(__file__).resolve().parents[1] / "scripts/train_first_stop.py").read_text(encoding="utf-8")
        self.assertIn('diagnostic_status="schedule_unavailable"', src)
        self.assertIn('diag = "processing_failure" if cached_locally else "cache_miss"', src)


if __name__ == "__main__":
    unittest.main()
