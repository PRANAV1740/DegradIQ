"""First-pit-stop prediction using historical race laps.

The model estimates a discrete hazard: at the end of each completed lap,
what is the probability that a driver's first pit stop occurs on the next lap?
It is a strategy forecast, not a causal tyre-wear model or guarantee.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
except ImportError:  # Vercel project root may be set to this package directory.
    from config import FULL_SEASON_DRIVERS, DRIVER_TEAM_2026

PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "first_stop_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "first_stop_model_metadata.json"

NUMERIC_FEATURES = [
    "LapNumber", "RaceLaps", "LapFraction", "TyreLife", "GridPosition",
    "Position", "RecentPaceLoss_s",
]
CATEGORICAL_FEATURES = ["Compound", "Driver", "Team", "EventName"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _to_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return float(pd.Timedelta(value).total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_valid(row: pd.Series, keys: tuple[str, ...], default: Any = np.nan) -> Any:
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            return row[key]
    return default


# FastF1 TrackStatus codes. Lap-level laps["TrackStatus"] concatenates every code
# seen during a lap (e.g. "451"), so it is NOT safe for red-flag classification;
# use the timestamped session.track_status timeline instead.
_RED_FLAG = "5"
# States under which an observation row may be used to predict the next lap, per the
# FROZEN target contract: green (1), Safety Car (4), VSC (6) and VSC-ending (7). Ordinary
# yellow (2) is NOT eligible and red (5) is excluded. Changing this set requires a
# separate documented contract review.
_ELIGIBLE_OBSERVATION_STATES = {"1", "4", "6", "7"}

# Label-definition version: identifies the reviewed target + label-generation semantics.
# Persisted in checkpoint metadata so training can refuse to mix incompatible label
# versions. Bump this whenever the label contract or generation logic changes.
LABEL_DEFINITION_VERSION = "2.0-strategic-green-vsc-sc"

# Extra (non-feature) output column flagging a missing/invalid lap time on an
# observation row. It is intentionally NOT part of FEATURE_COLUMNS so it cannot
# change the input schema of the already-trained production/candidate models; a
# future retrain may promote it into the feature set.
EXTRA_OUTPUT_COLUMNS = ["StopNextLap", "LapTimeMissing"]


def _build_status_intervals(track_status: pd.DataFrame | None) -> list[tuple[Any, str]]:
    """Return sorted (session_time, status_code) changes from a track_status frame.

    Times are session-relative (Timedelta). Returns [] when the timeline is
    unavailable so callers fall back to a conservative, clearly-marked path.
    """
    if track_status is None or len(track_status) == 0:
        return []
    if "Time" not in track_status.columns or "Status" not in track_status.columns:
        return []
    ts = track_status.dropna(subset=["Time"]).sort_values("Time")
    return [(r["Time"], str(r["Status"])) for _, r in ts.iterrows()]


def _confirmed_state_at(intervals: list[tuple[Any, str]], t: Any) -> str | None:
    """Confirmed track-status code active at session-time t, or None if unresolved.

    Deterministic: the last status change at or before t wins. None means the
    state cannot be established (no timeline, or t missing) and must not be guessed.
    """
    if not intervals or t is None or pd.isna(t):
        return None
    code = None
    for start, status in intervals:
        if start <= t:
            code = status
        else:
            break
    return code


def build_hazard_rows(laps: pd.DataFrame, results: pd.DataFrame | None = None,
                      event_name: str = "Unknown", race_laps: int | None = None,
                      track_status: pd.DataFrame | None = None,
                      diagnostics_sink: list | None = None) -> pd.DataFrame:
    """Convert raw FastF1 race laps into next-lap first-stop hazard examples.

    Target contract (frozen): predict each driver's first genuine strategic
    tyre-changing stop. Discretionary stops under green, VSC or Safety Car are
    in scope; drive-throughs and pass-throughs without a real tyre change are
    excluded, as are pit entries confirmed (by timestamp) to occur during an
    active red flag, and the existing lap<=2 SC/red compulsory-stop exclusion.

    Horizon: an observation at the end of lap k predicts whether the first
    strategic pit entry occurs on lap k+1. Positive labels are anchored to the
    actual pit-entry lap number; they are never relocated to an earlier lap.

    Event detection is kept strictly separate from feature-row eligibility:
      * Pit events are detected from raw timing/stint/compound/tyre-age evidence
        BEFORE any lap-time validity filtering, so a pit-entry lap with a missing
        LapTime (e.g. Magnussen, Austria 2023) does not erase the event.
      * Observation feature rows are emitted only for laps strictly before the
        first target stop; no lap at or after the stop is ever emitted.

    Eligible observation states are green, Safety Car and VSC (codes 1/4/6/7);
    ordinary yellow (2) and red (5) are not eligible. `track_status` is the
    timestamped session.track_status frame; when absent, states cannot be
    confirmed and the function falls back to a documented green-only rule
    (`timeline_available=False` in diagnostics), never classifying from the
    concatenated lap status string.

    Diagnostics: a structured per-event record (event, label_definition_version,
    timeline_available, unresolved_count, unresolved_events, reason) is appended to
    `diagnostics_sink` when provided — a reliable channel that, unlike
    `DataFrame.attrs`, survives aggregation. The record is also mirrored on
    `.attrs["diagnostics"]`/`.attrs["unresolved_events"]` for single-frame use.
    """
    def _empty(reason: str = "no_rows") -> pd.DataFrame:
        out = pd.DataFrame(columns=FEATURE_COLUMNS + EXTRA_OUTPUT_COLUMNS)
        out.attrs["unresolved_events"] = []
        diag = {"event": str(event_name), "label_definition_version": LABEL_DEFINITION_VERSION,
                "timeline_available": None, "unresolved_count": 0, "unresolved_events": [], "reason": reason}
        out.attrs["diagnostics"] = diag
        if diagnostics_sink is not None:
            diagnostics_sink.append(diag)
        return out

    if laps is None or laps.empty:
        return _empty("empty_laps")
    df = laps.copy()
    required = {"Driver", "LapNumber", "LapTime", "Compound", "TyreLife"}
    if not required.issubset(df.columns):
        return _empty("missing_required_columns")

    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df["LapTime_s"] = df["LapTime"].map(_to_seconds)
    # DETECTION frame: keep rows even when LapTime is missing/invalid. The lap-time
    # validity gate is applied per-feature later, not as a row filter, so a pit-entry
    # lap with NaT LapTime survives for event detection and next-lap adjacency.
    df = df.dropna(subset=["Driver", "LapNumber", "TyreLife", "Compound"]).copy()
    if "PitInTime" not in df.columns:
        df["PitInTime"] = pd.NaT
    if "Position" not in df.columns:
        df["Position"] = np.nan
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    # Match grid position/team from session results when available.
    result_map: dict[str, dict[str, Any]] = {}
    if results is not None and not results.empty:
        result_driver_col = "Abbreviation" if "Abbreviation" in results.columns else (
            "Driver" if "Driver" in results.columns else None)
        if result_driver_col:
            for _, r in results.iterrows():
                code = str(r.get(result_driver_col, ""))
                if not code or code == "nan":
                    continue
                result_map[code] = {
                    "GridPosition": _first_valid(r, ("GridPosition", "Grid")),
                    "Team": _first_valid(r, ("TeamName", "Team"), "Unknown"),
                }

    if race_laps is None:
        race_laps = int(df["LapNumber"].max()) if not df.empty else 0
    if race_laps <= 0:
        return _empty("no_race_distance")

    intervals = _build_status_intervals(track_status)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for driver, group in df.groupby("Driver", sort=False):
        g = group.sort_values("LapNumber").drop_duplicates("LapNumber", keep="last").reset_index(drop=True)
        if len(g) < 2:
            continue

        # ---------------- EVENT DETECTION (raw; before lap-time filtering) ----------------
        # Step through pit entries in lap order; take the first genuine strategic
        # tyre-changing candidate, classifying its CONFIRMED state at the pit-entry
        # timestamp. Red-flag changes are skipped (not strategic); a candidate whose
        # entry state cannot be confirmed is kept as an UNRESOLVED stop (fail-safe:
        # it truncates the risk set but produces no label).
        first_pit_lap = None
        first_pit_resolved = True
        if "Stint" in g.columns:
            # Candidate pit-entry laps come from a PitInTime OR a stint increase on the
            # next lap (a tyre change can leave stint/compound evidence even when the
            # pit-entry timestamp is missing). Iterate in lap order.
            pit_present = g["PitInTime"].notna().to_numpy()
            stint_vals = g["Stint"].to_numpy()
            candidate_idx = []
            for i in range(len(g) - 1):
                nxt = pd.notna(stint_vals[i]) and pd.notna(stint_vals[i + 1]) and stint_vals[i] != stint_vals[i + 1]
                if pit_present[i] or nxt:
                    candidate_idx.append(i)

            for pit_idx in candidate_idx:
                lap_num = float(g.iloc[pit_idx]["LapNumber"])
                t_status_str = str(g.iloc[pit_idx].get("TrackStatus", ""))
                old_c = str(g.iloc[pit_idx].get("Compound", "")).upper()
                new_c = str(g.iloc[pit_idx + 1].get("Compound", "")).upper()
                old_age = g.iloc[pit_idx].get("TyreLife", np.nan)
                new_age = g.iloc[pit_idx + 1].get("TyreLife", np.nan)
                has_pit_time = pd.notna(g.iloc[pit_idx].get("PitInTime", pd.NaT))

                # Preserve existing compulsory opening-lap exclusion (lap<=2 under SC/red).
                if lap_num <= 2 and ("5" in t_status_str or "4" in t_status_str):
                    continue

                # Genuine tyre change: stint or compound change on the out-lap. The new
                # stint's compound/tyre age legitimately appear on the out-lap (pit_idx+1),
                # not on the pit-entry row; the pit-entry lap remains the target event.
                old_stint, new_stint = g.iloc[pit_idx]["Stint"], g.iloc[pit_idx + 1]["Stint"]
                stint_changed = pd.notna(old_stint) and pd.notna(new_stint) and old_stint != new_stint
                compound_changed = bool(old_c and new_c and old_c != new_c and old_c != "UNKNOWN" and new_c != "UNKNOWN")
                tyre_reset = bool(pd.notna(new_age) and pd.notna(old_age) and new_age < old_age)
                if not (stint_changed or compound_changed):
                    continue  # drive-through / pass-through, no tyre change
                if not compound_changed and not tyre_reset and ("4" in t_status_str or "5" in t_status_str):
                    continue  # pass-through under caution with no tyre evidence

                # Classify the CONFIRMED state at the pit-entry timestamp (never via the
                # concatenated lap status string). An unconfirmable state (no usable
                # PitInTime, or a timestamp the timeline does not cover) is NOT evidence
                # that no red flag was active -> fail safe to an UNRESOLVED stop.
                if intervals:
                    entry_state = _confirmed_state_at(intervals, g.iloc[pit_idx].get("PitInTime", pd.NaT)) if has_pit_time else None
                    if entry_state == _RED_FLAG:
                        continue  # confirmed red-flag change: not strategic; try the next pit
                    if entry_state is None:
                        first_pit_lap = lap_num
                        first_pit_resolved = False
                        unresolved.append({"driver": str(driver), "event": str(event_name), "stop_lap": lap_num,
                                           "reason": "pit_entry_state_unconfirmed" if has_pit_time
                                                     else "pit_entry_timestamp_missing"})
                        break
                    first_pit_lap = lap_num  # confirmed non-red eligible state
                    first_pit_resolved = True
                    break
                else:
                    # No timestamped timeline: red cannot be confirmed. Documented fallback
                    # (event-level `timeline_available=False`): accept the candidate and apply
                    # green-only observation eligibility below.
                    first_pit_lap = lap_num
                    first_pit_resolved = True
                    break

        meta = result_map.get(str(driver), {})
        grid = pd.to_numeric(pd.Series([meta.get("GridPosition", np.nan)]), errors="coerce").iloc[0]
        team = meta.get("Team", "Unknown")
        if pd.isna(team):
            team = "Unknown"

        times = g["LapTime_s"].to_numpy(float)
        compounds = g["Compound"].astype(str).to_numpy()
        tyre_ages = g["TyreLife"].to_numpy(float)
        lap_nums = g["LapNumber"].to_numpy(float)
        positions = g["Position"].to_numpy(float)
        stints = g["Stint"].to_numpy() if "Stint" in g.columns else np.zeros(len(g))
        lap_set = set(float(x) for x in lap_nums)

        # If a target stop exists but its predicting lap (first_pit_lap - 1) is absent
        # from the frame, the one-step observation cannot be established. Do not relocate
        # the label; record the event as unresolved (the driver is censored here).
        if first_pit_lap is not None and first_pit_resolved and (first_pit_lap - 1) not in lap_set:
            first_pit_resolved = False
            unresolved.append({"driver": str(driver), "event": str(event_name),
                               "stop_lap": first_pit_lap, "reason": "predicting_row_absent"})

        def _flag_unresolved(next_lap: float, reason: str) -> None:
            if first_pit_lap is not None and next_lap == first_pit_lap:
                unresolved.append({"driver": str(driver), "event": str(event_name),
                                   "stop_lap": first_pit_lap, "reason": reason})

        for i in range(len(g) - 1):
            current_lap = lap_nums[i]
            next_lap = lap_nums[i + 1]
            # Truncate the first-stop risk set at the actual stop lap: never emit a lap
            # at or after the driver's first target stop.
            if first_pit_lap is not None and current_lap >= first_pit_lap:
                break
            if not np.isfinite(current_lap) or next_lap != current_lap + 1:
                continue
            # Unresolved stop: never emit a guessed positive OR negative for the row
            # that would predict it (the unresolved record was already logged in detection).
            if first_pit_lap is not None and not first_pit_resolved and next_lap == first_pit_lap:
                continue

            row = g.iloc[i]
            # Observation-state eligibility at the end of lap k (the cutoff).
            if intervals:
                obs_state = _confirmed_state_at(intervals, row.get("Time", pd.NaT))
                if obs_state is None:
                    _flag_unresolved(next_lap, "observation_state_unresolved")
                    continue
                if obs_state == _RED_FLAG or obs_state not in _ELIGIBLE_OBSERVATION_STATES:
                    _flag_unresolved(next_lap, f"predicting_lap_state_{obs_state}")
                    continue
            else:
                # No timestamped timeline: cannot confirm VSC/SC; fall back to green-only.
                if "TrackStatus" in g.columns and str(row.get("TrackStatus")) != "1":
                    _flag_unresolved(next_lap, "predicting_lap_nongreen_no_timeline")
                    continue

            if "IsAccurate" in g.columns and pd.notna(row.get("IsAccurate")) and not bool(row.get("IsAccurate")):
                _flag_unresolved(next_lap, "predicting_lap_inaccurate")
                continue

            # Baseline only uses laps from the current stint and never future laps.
            same_stint_indices = [j for j in range(i + 1) if stints[j] == stints[i]]
            recent_indices = same_stint_indices[-3:]
            baseline_indices = same_stint_indices[:min(3, len(same_stint_indices))]
            recent_median = float(np.nanmedian(times[recent_indices])) if recent_indices else np.nan
            baseline_median = float(np.nanmedian(times[baseline_indices])) if baseline_indices else np.nan
            pace_loss = recent_median - baseline_median if np.isfinite(recent_median) and np.isfinite(baseline_median) else np.nan
            lap_time_missing = int(not (np.isfinite(times[i]) and times[i] > 0))
            # Label anchored to the ACTUAL pit-entry lap number, not a row position.
            target = int(first_pit_lap is not None and next_lap == first_pit_lap)
            rows.append({
                "LapNumber": float(current_lap),
                "RaceLaps": float(race_laps),
                "LapFraction": float(current_lap / race_laps),
                "TyreLife": float(tyre_ages[i]),
                "GridPosition": float(grid) if pd.notna(grid) else np.nan,
                "Position": float(positions[i]) if np.isfinite(positions[i]) else (float(grid) if pd.notna(grid) else np.nan),
                "RecentPaceLoss_s": float(pace_loss) if np.isfinite(pace_loss) else np.nan,
                "Compound": compounds[i].upper(),
                "Driver": str(driver),
                "Team": str(team),
                "EventName": str(event_name),
                "StopNextLap": target,
                "LapTimeMissing": lap_time_missing,
            })

    out = pd.DataFrame(rows, columns=FEATURE_COLUMNS + EXTRA_OUTPUT_COLUMNS)
    diagnostics = {
        "event": str(event_name),
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "timeline_available": bool(intervals),
        "unresolved_count": len(unresolved),
        "unresolved_events": unresolved,
        "reason": "ok",
    }
    out.attrs["unresolved_events"] = unresolved
    out.attrs["diagnostics"] = diagnostics
    if diagnostics_sink is not None:
        diagnostics_sink.append(diagnostics)
    return out


def fit_first_stop_model(training_rows: pd.DataFrame) -> dict[str, Any]:
    """Fit the discrete hazard model and return training diagnostics."""
    if training_rows is None or training_rows.empty:
        raise ValueError("No first-stop training rows were provided.")
    data = training_rows.dropna(subset=["StopNextLap"]).copy()
    data["StopNextLap"] = pd.to_numeric(data["StopNextLap"], errors="coerce")
    data = data.dropna(subset=["StopNextLap"])
    positives = int((data["StopNextLap"] == 1).sum())
    negatives = int((data["StopNextLap"] == 0).sum())
    if positives < 10 or negatives < 100:
        raise ValueError(f"Insufficient training data: {positives} positive stop events and {negatives} non-stop rows; need at least 10 and 100.")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("hazard", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
    ])
    pipeline.fit(data[FEATURE_COLUMNS], data["StopNextLap"].astype(int))
    return {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(data)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_rate": float(positives / len(data)),
    }


def save_first_stop_model(model_bundle: dict[str, Any], model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path, compress=3)
    metadata = {k: v for k, v in model_bundle.items() if k != "pipeline"}
    metadata["model_file"] = model_path.name
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_first_stop_model(model_path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def _lap_features(driver: dict[str, Any], lap_number: int, race_laps: int,
                  event_name: str, mode: str) -> dict[str, Any]:
    if mode == "pre-race":
        current_lap = lap_number - 1
        tyre_age = float(driver.get("starting_tyre_age", 1)) + current_lap - 1
        position = driver.get("grid_position")
        pace_loss = np.nan
        compound = driver.get("starting_compound", "MEDIUM")
    else:
        current_lap = lap_number - 1
        elapsed = max(0, current_lap - int(driver.get("current_lap", 0)))
        tyre_age = float(driver.get("current_tyre_age", driver.get("starting_tyre_age", 1))) + elapsed
        position = driver.get("current_position", driver.get("grid_position"))
        pace_loss = driver.get("recent_pace_loss_s", np.nan)
        compound = driver.get("current_compound", driver.get("starting_compound", "MEDIUM"))
    return {
        "LapNumber": float(current_lap),
        "RaceLaps": float(race_laps),
        "LapFraction": float(current_lap / race_laps),
        "TyreLife": max(1.0, float(tyre_age)),
        "GridPosition": float(driver["grid_position"]) if driver.get("grid_position") is not None else np.nan,
        "Position": float(position) if position is not None else np.nan,
        "RecentPaceLoss_s": float(pace_loss) if pace_loss is not None else np.nan,
        "Compound": str(compound).upper(),
        "Driver": str(driver["driver"]).upper(),
        "Team": str(driver.get("team") or DRIVER_TEAM_2026.get(str(driver["driver"]).upper(), "Unknown")),
        "EventName": str(event_name),
    }


def predict_first_stops(model_bundle: dict[str, Any], event_name: str, race_laps: int,
                        drivers: list[dict[str, Any]], mode: str = "pre-race") -> dict[str, Any]:
    """Forecast each driver's first stop from per-lap conditional hazard scores.

    `mode='pre-race'` uses starting grid/compound/tyre-age inputs only.
    `mode='live'` also uses current lap, tyre age/compound, position and recent pace loss.
    """
    if mode not in {"pre-race", "live"}:
        raise ValueError("mode must be 'pre-race' or 'live'.")
    if race_laps < 5:
        raise ValueError("race_laps must be at least 5.")
    if not drivers:
        raise ValueError("At least one driver is required.")
    pipeline = model_bundle["pipeline"]
    output = []
    for driver in drivers:
        code = str(driver.get("driver", "")).upper()
        if code not in FULL_SEASON_DRIVERS:
            continue
        if driver.get("grid_position") is None:
            continue
        if mode == "live" and driver.get("current_lap") is None:
            continue
        first_candidate = 2 if mode == "pre-race" else max(2, int(driver.get("current_lap", 0)) + 1)
        if first_candidate > race_laps:
            output.append({"driver": code, "status": "race distance already reached", "probability_stop_before_finish": 0.0})
            continue
        feature_rows = [
            _lap_features(driver, lap, race_laps, event_name, mode)
            for lap in range(first_candidate, race_laps + 1)
        ]
        frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        hazards = pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        hazards = np.clip(hazards, 0.0, 1.0)
        survival = 1.0
        lap_probabilities = []
        cumulative = []
        for lap, hazard in zip(range(first_candidate, race_laps + 1), hazards):
            stop_prob = survival * float(hazard)
            survival *= 1.0 - float(hazard)
            lap_probabilities.append({"lap": int(lap), "probability": stop_prob})
            cumulative.append({"lap": int(lap), "probability_stopped_by_lap": 1.0 - survival})
        total_stop_prob = 1.0 - survival
        conditional = np.asarray([x["probability"] for x in lap_probabilities], dtype=float)
        if conditional.sum() > 0:
            conditional /= conditional.sum()
            cdf = np.cumsum(conditional)
            lap_values = np.asarray([x["lap"] for x in lap_probabilities])
            q10 = int(lap_values[min(np.searchsorted(cdf, 0.10), len(lap_values) - 1)])
            q50 = int(lap_values[min(np.searchsorted(cdf, 0.50), len(lap_values) - 1)])
            q90 = int(lap_values[min(np.searchsorted(cdf, 0.90), len(lap_values) - 1)])
            expected = float(np.sum(lap_values * conditional))
            most_likely = int(lap_values[int(np.argmax(conditional))])
        else:
            q10 = q50 = q90 = most_likely = None
            expected = None
        output.append({
            "driver": code,
            "team": driver.get("team") or DRIVER_TEAM_2026.get(code, "Unknown"),
            "status": "forecast",
            "probability_stop_before_finish": round(float(total_stop_prob), 4),
            "most_likely_stop_lap": most_likely,
            "median_stop_lap_if_stops": q50,
            "likely_window_if_stops": {"from_lap": q10, "to_lap": q90},
            "expected_stop_lap_if_stops": round(expected, 2) if expected is not None else None,
            "lap_probabilities": lap_probabilities,
            "cumulative_probabilities": cumulative,
        })
    output.sort(key=lambda x: (x.get("expected_stop_lap_if_stops") is None,
                               x.get("expected_stop_lap_if_stops") or float("inf")))
    for index, item in enumerate(output, start=1):
        item["predicted_order_by_expected_lap"] = index
    return {
        "season": 2026,
        "event": event_name,
        "mode": mode,
        "race_laps": race_laps,
        "model_type": "discrete first-stop hazard (logistic regression)",
        "interpretation": "Stop-lap probabilities are conditional-hazard forecasts. The reported window is conditional on stopping before race distance; it is not a guarantee or an optimal-strategy claim.",
        "driver_count_forecast": len(output),
        "predictions": output,
    }
