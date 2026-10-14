"""Stage 6 support: honest practice-to-race holdout validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .pipeline import defuel, detraffic, deevolve
from .model import fit_hinge, predict


def prepare_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same physical corrections and filters to practice or race laps."""
    if df.empty:
        return df.copy()
    return deevolve(detraffic(defuel(df)))


def _baseline_delta(group: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = group.sort_values("TyreLife").copy()
    baseline = g.loc[g["TyreLife"] == 2, value_col]
    if baseline.empty:
        baseline = g[value_col].iloc[:1]
    g["ActualDelta_s"] = g[value_col] - float(baseline.iloc[0])
    return g


def validate_compound(practice_clean: pd.DataFrame, race_clean: pd.DataFrame,
                      compound: str) -> dict:
    """Fit on practice only, then score matching race laps relative to each stint's lap 2."""
    train = practice_clean[practice_clean["Compound"] == compound].copy()
    race = race_clean[race_clean["Compound"] == compound].copy()
    model_info = fit_hinge(train)
    if model_info.get("status") != "fit":
        return {"compound": compound, "status": model_info.get("status"),
                "mae_s_per_lap": None, "race_laps": 0, "stints": 0, "model": model_info}
    if race.empty:
        return {"compound": compound, "status": "no race laps", "mae_s_per_lap": None,
                "race_laps": 0, "stints": 0, "model": model_info}

    predictions, actuals, ages_all = [], [], []
    valid_stint_count = 0
    stint_keys = [c for c in ("EventName", "Driver", "Stint") if c in race.columns]
    for _, stint in race.groupby(stint_keys, dropna=False):
        stint = stint.dropna(subset=["TyreLife", "CorrectedTime_s"]).sort_values("TyreLife").copy()
        if stint.empty:
            continue
        baseline = stint.loc[stint["TyreLife"] == 2, "CorrectedTime_s"]
        # TyreLife >= 4 filtering means age 2 is often absent. In that case,
        # use the earliest eligible age and make this fallback explicit in reports.
        baseline_age = 2.0
        if baseline.empty:
            baseline_age = float(stint["TyreLife"].iloc[0])
            baseline = stint["CorrectedTime_s"].iloc[:1]
        actual_delta = stint["CorrectedTime_s"].to_numpy(float) - float(baseline.iloc[0])
        pred_delta = predict(model_info, stint["TyreLife"].to_numpy(float)) - predict(
            model_info, np.array([baseline_age])
        )[0]
        valid = np.isfinite(actual_delta) & np.isfinite(pred_delta)
        if valid.any():
            valid_stint_count += 1
        predictions.extend(pred_delta[valid].tolist())
        actuals.extend(actual_delta[valid].tolist())
        ages_all.extend(stint.loc[valid, "TyreLife"].astype(float).tolist())

    if not actuals:
        return {"compound": compound, "status": "no comparable race laps", "mae_s_per_lap": None,
                "race_laps": 0, "stints": 0, "model": model_info}
    return {"compound": compound, "status": "validated", "mae_s_per_lap": float(
                mean_absolute_error(actuals, predictions)),
            "race_laps": len(actuals), "stints": valid_stint_count, "model": model_info,
            "ages": ages_all, "actual_delta": actuals, "predicted_delta": predictions}


def validate_event(practice_df: pd.DataFrame, race_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return per-compound validation rows and chart-ready holdout data."""
    practice = prepare_laps(practice_df)
    race = prepare_laps(race_df)
    rows, details = [], {}
    compounds = sorted(set(practice.get("Compound", [])) | set(race.get("Compound", [])))
    for compound in compounds:
        result = validate_compound(practice, race, compound)
        details[compound] = result
        rows.append({k: result.get(k) for k in ("compound", "status", "mae_s_per_lap", "race_laps", "stints")})
    return pd.DataFrame(rows), details
