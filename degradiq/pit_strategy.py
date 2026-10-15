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


def build_hazard_rows(laps: pd.DataFrame, results: pd.DataFrame | None = None,
                      event_name: str = "Unknown", race_laps: int | None = None) -> pd.DataFrame:
    """Convert raw FastF1 race laps into next-lap first-stop hazard examples.

    Features at lap k predict whether the driver's first pit-in occurs on lap k+1.
    Rows after the first stop are excluded. The final observed lap is excluded
    because there is no next-lap outcome to label.
    """
    if laps is None or laps.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["StopNextLap"])
    df = laps.copy()
    required = {"Driver", "LapNumber", "LapTime", "Compound", "TyreLife"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["StopNextLap"])

    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df["LapTime_s"] = df["LapTime"].map(_to_seconds)
    df = df.dropna(subset=["Driver", "LapNumber", "TyreLife", "LapTime_s", "Compound"])
    df = df[df["LapTime_s"] > 0].copy()
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
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["StopNextLap"])

    rows: list[dict[str, Any]] = []
    for driver, group in df.groupby("Driver", sort=False):
        g = group.sort_values("LapNumber").drop_duplicates("LapNumber", keep="last").reset_index(drop=True)
        if len(g) < 2:
            continue
        pit_flags = g["PitInTime"].notna().to_numpy()
        # Require evidence that the pit visit is a genuine strategic tyre stop:
        # 1. Distinguish strategic stops from compulsory red-flag race suspensions or safety car pass-throughs on Lap <= 2.
        # 2. Exclude non-tyre changes (safety car pit-lane pass-throughs, drive-throughs) where compound didn't change and tyre life didn't reset.
        if "Stint" in g.columns:
            for pit_idx in np.flatnonzero(pit_flags):
                if pit_idx + 1 >= len(g):
                    pit_flags[pit_idx] = False
                    continue
                lap_num = g.iloc[pit_idx]["LapNumber"]
                t_status = str(g.iloc[pit_idx].get("TrackStatus", ""))
                old_c = str(g.iloc[pit_idx].get("Compound", "")).upper()
                new_c = str(g.iloc[pit_idx + 1].get("Compound", "")).upper()
                old_age = g.iloc[pit_idx].get("TyreLife", np.nan)
                new_age = g.iloc[pit_idx + 1].get("TyreLife", np.nan)

                # Compulsory race suspension or safety-car routing on Lap 1 or 2:
                if lap_num <= 2 and ("5" in t_status or "4" in t_status):
                    pit_flags[pit_idx] = False
                    continue

                old_stint, new_stint = g.iloc[pit_idx]["Stint"], g.iloc[pit_idx + 1]["Stint"]
                stint_changed = pd.notna(old_stint) and pd.notna(new_stint) and old_stint != new_stint
                compound_changed = bool(old_c and new_c and old_c != new_c and old_c != "UNKNOWN" and new_c != "UNKNOWN")
                tyre_reset = bool(pd.notna(new_age) and pd.notna(old_age) and new_age < old_age)

                if not (stint_changed or compound_changed):
                    pit_flags[pit_idx] = False
                elif not compound_changed and not tyre_reset and ("4" in t_status or "5" in t_status):
                    pit_flags[pit_idx] = False
        pit_indices = np.flatnonzero(pit_flags)
        first_pit_idx = int(pit_indices[0]) if len(pit_indices) else None
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

        for i in range(len(g) - 1):
            # Only pre-first-stop states are training examples.
            if first_pit_idx is not None and i >= first_pit_idx:
                break
            current_lap = lap_nums[i]
            next_lap = lap_nums[i + 1]
            if not np.isfinite(current_lap) or next_lap != current_lap + 1:
                continue
            # Avoid training on inaccurate/yellow-flag state laps when those flags exist.
            row = g.iloc[i]
            if "IsAccurate" in g.columns and pd.notna(row.get("IsAccurate")) and not bool(row.get("IsAccurate")):
                continue
            if "TrackStatus" in g.columns and str(row.get("TrackStatus")) != "1":
                continue
            # Baseline only uses laps from the current stint and never future laps.
            same_stint_indices = [j for j in range(i + 1) if stints[j] == stints[i]]
            recent_indices = same_stint_indices[-3:]
            baseline_indices = same_stint_indices[:min(3, len(same_stint_indices))]
            recent_median = float(np.nanmedian(times[recent_indices])) if recent_indices else np.nan
            baseline_median = float(np.nanmedian(times[baseline_indices])) if baseline_indices else np.nan
            pace_loss = recent_median - baseline_median if np.isfinite(recent_median) and np.isfinite(baseline_median) else np.nan
            target = int(bool(pit_flags[i + 1]))
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
            })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["StopNextLap"])


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
