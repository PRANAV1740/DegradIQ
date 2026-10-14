"""Download completed historical races, validate chronologically, and train the first-stop model.

Run from repository root:
    python -m degradiq.scripts.train_first_stop --start-year 2022 --end-year 2026

This is an offline/local task, not a Vercel request handler. It writes artifacts and
reports only inside `degradiq/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.config import YEAR
from degradiq.pit_strategy import (
    PACKAGE_DIR, FEATURE_COLUMNS, build_hazard_rows, fit_first_stop_model,
    save_first_stop_model, _lap_features,
)

CACHE = PACKAGE_DIR / "cache" / "fastf1"
REPORT_DIR = PACKAGE_DIR / "reports"


def _event_date(row: pd.Series):
    value = row.get("EventDate", row.get("Session5Date", pd.NaT))
    return pd.to_datetime(value, errors="coerce", utc=True)


def collect_rows(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    all_rows, statuses = [], []
    now = pd.Timestamp.now(tz="UTC")
    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            statuses.append({"year": year, "event": "*", "status": "schedule unavailable", "rows": 0,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        for _, event in schedule.iterrows():
            name = str(event.get("EventName", "Unknown"))
            event_date = _event_date(event)
            if pd.notna(event_date) and event_date > now:
                statuses.append({"year": year, "event": name, "status": "future/pending", "rows": 0, "error": ""})
                continue
            event_id = event.get("RoundNumber", name)
            if pd.isna(event_id) or event_id == 0:
                event_id = name
            try:
                session = fastf1.get_session(year, event_id, "R")
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
                results = getattr(session, "results", pd.DataFrame())
                max_lap = pd.to_numeric(laps.get("LapNumber", pd.Series(dtype=float)), errors="coerce").max()
                rows = build_hazard_rows(laps, results, name, int(max_lap) if pd.notna(max_lap) else None)
                if not rows.empty:
                    rows["Season"] = year
                    all_rows.append(rows)
                statuses.append({"year": year, "event": name, "status": "loaded" if not rows.empty else "no eligible rows",
                                 "rows": len(rows), "positive_rows": int(rows.get("StopNextLap", pd.Series(dtype=int)).sum()), "error": ""})
                print(f"{year} {name}: {len(rows)} rows, {int(rows.get('StopNextLap', pd.Series(dtype=int)).sum())} stop labels")
            except Exception as exc:
                statuses.append({"year": year, "event": name, "status": "unavailable", "rows": 0,
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"Skip {year} {name}: {type(exc).__name__}: {exc}")
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return combined, pd.DataFrame(statuses)


def chronological_backtest(rows: pd.DataFrame, holdout_year: int) -> dict:
    """Fit on seasons strictly before holdout_year and score the untouched holdout season."""
    train = rows[rows["Season"] < holdout_year].copy()
    test = rows[rows["Season"] == holdout_year].copy()
    if train.empty or test.empty:
        return {"status": "not available", "reason": f"Need both pre-{holdout_year} training rows and {holdout_year} holdout rows.",
                "train_rows": len(train), "holdout_rows": len(test)}
    try:
        bundle = fit_first_stop_model(train)
    except ValueError as exc:
        return {"status": "not available", "reason": str(exc), "train_rows": len(train), "holdout_rows": len(test)}
    X_test = test[FEATURE_COLUMNS]
    y_test = test["StopNextLap"].astype(int).to_numpy()
    p = bundle["pipeline"].predict_proba(X_test)[:, 1]
    result = {
        "status": "evaluated",
        "holdout_year": int(holdout_year),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "holdout_positive_rows": int(y_test.sum()),
        "brier_score": float(brier_score_loss(y_test, p)),
        "log_loss": float(log_loss(y_test, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) == 2 else None,
        "observed_stop_rate": float(y_test.mean()),
        "mean_predicted_hazard": float(np.mean(p)),
        "first_stop_lap_mae": None,
        "drivers_with_observed_first_stop": 0,
    }
    # Group-level timing check. Simulate the full candidate-lap horizon from
    # pre-race inputs; do not truncate the forecast at the actual stop lap.
    # The actual first-stop lap is used only as the evaluation target.
    errors = []
    for (_, _), group in test.groupby(["EventName", "Driver"], dropna=False):
        group = group.sort_values("LapNumber")
        positives = group[group["StopNextLap"] == 1]
        if positives.empty:
            continue  # censored/no observed first stop: not scored for lap error
        first = group.iloc[0]
        driver_state = {
            "driver": str(first["Driver"]),
            "team": str(first.get("Team", "Unknown")),
            "grid_position": first.get("GridPosition"),
            "starting_compound": str(first.get("Compound", "MEDIUM")),
            "starting_tyre_age": max(1, int(first.get("TyreLife", 1))),
        }
        race_distance = max(5, int(first.get("RaceLaps", group["LapNumber"].max())))
        forecast_rows = [
            _lap_features(driver_state, lap, race_distance, str(first["EventName"]), "pre-race")
            for lap in range(2, race_distance + 1)
        ]
        if not forecast_rows:
            continue
        group_p = bundle["pipeline"].predict_proba(pd.DataFrame(forecast_rows)[FEATURE_COLUMNS])[:, 1]
        survival = 1.0
        masses = []
        laps = np.arange(2, race_distance + 1)
        for prob in group_p:
            masses.append(survival * float(prob))
            survival *= 1.0 - float(prob)
        masses = np.asarray(masses, dtype=float)
        if masses.sum() <= 0:
            continue
        masses /= masses.sum()  # conditional on a first stop within race distance
        predicted_lap = int(laps[min(np.searchsorted(np.cumsum(masses), 0.5), len(laps) - 1)])
        actual_lap = int(positives["LapNumber"].iloc[0]) + 1
        errors.append(abs(predicted_lap - actual_lap))
    result["drivers_with_observed_first_stop"] = len(errors)
    result["first_stop_lap_mae"] = float(np.mean(errors)) if errors else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=YEAR)
    parser.add_argument("--holdout-year", type=int, default=2025,
                        help="Whole season reserved for chronological backtest before final refit.")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")

    rows, coverage = collect_rows(args.start_year, args.end_year)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(REPORT_DIR / "first_stop_training_coverage.csv", index=False)
    if rows.empty:
        raise SystemExit("No historical first-stop training rows were available. Check FastF1 access and coverage report.")

    # Keep training examples as an auditable local report; no generated file escapes degradiq/.
    rows.to_csv(REPORT_DIR / "first_stop_training_rows.csv", index=False)
    validation = chronological_backtest(rows, args.holdout_year)
    (REPORT_DIR / "first_stop_backtest.json").write_text(pd.Series(validation).to_json(indent=2), encoding="utf-8")
    print("Chronological holdout:")
    print(pd.Series(validation).to_string())

    # Refit the deployable artifact on all completed race data after the holdout result is recorded.
    bundle = fit_first_stop_model(rows)
    bundle["training_seasons"] = sorted(int(y) for y in rows["Season"].dropna().unique())
    bundle["chronological_backtest"] = validation
    save_first_stop_model(bundle)
    print(f"Model artifact: {PACKAGE_DIR / 'artifacts' / 'first_stop_model.joblib'}")
    print(f"Metadata: {PACKAGE_DIR / 'artifacts' / 'first_stop_model_metadata.json'}")
    print(f"Coverage: {REPORT_DIR / 'first_stop_training_coverage.csv'}")


if __name__ == "__main__":
    main()
