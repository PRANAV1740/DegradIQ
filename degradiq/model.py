"""Stage 5: reproducible, non-negative piecewise-linear hinge degradation model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, MIN_MODEL_ROWS, MIN_UNIQUE_TYRE_AGES


def hinge_value(age, intercept, early_slope, late_extra_slope, knot):
    x = np.asarray(age, dtype=float)
    return intercept + early_slope * x + late_extra_slope * np.maximum(0.0, x - knot)


def fit_hinge(df: pd.DataFrame) -> dict:
    """Fit a continuous hinge model by deterministic knot search and least squares."""
    clean = df.dropna(subset=["TyreLife", "CorrectedTime_s"]).copy()
    clean["TyreLife"] = pd.to_numeric(clean["TyreLife"], errors="coerce")
    clean["CorrectedTime_s"] = pd.to_numeric(clean["CorrectedTime_s"], errors="coerce")
    clean = clean.dropna(subset=["TyreLife", "CorrectedTime_s"])
    clean = clean[np.isfinite(clean["TyreLife"]) & np.isfinite(clean["CorrectedTime_s"])]
    ages = clean["TyreLife"].to_numpy(float)
    times = clean["CorrectedTime_s"].to_numpy(float)
    if len(clean) < MIN_MODEL_ROWS or np.unique(ages).size < MIN_UNIQUE_TYRE_AGES:
        return {"status": "insufficient data", "n_laps": int(len(clean)), "model": None}
    # Fixed seed retained for reproducibility if stochastic fitting is added later.
    rng = np.random.RandomState(RANDOM_STATE)
    knots = np.unique(np.quantile(ages, np.linspace(0.25, 0.75, 11)))
    best = None
    for knot in knots:
        design = np.column_stack([np.ones_like(ages), ages, np.maximum(0.0, ages-knot)])
        coef, *_ = np.linalg.lstsq(design, times, rcond=None)
        # Enforce nonnegative early wear and nonnegative additional late wear.
        coef[1] = max(0.0, float(coef[1]))
        coef[2] = max(0.0, float(coef[2]))
        # Refit intercept only after clamping slopes.
        coef[0] = float(np.mean(times - (coef[1]*ages + coef[2]*np.maximum(0.0, ages-knot))))
        pred = design @ coef
        mae = float(np.mean(np.abs(times-pred)))
        if best is None or mae < best["train_mae_s"]:
            best = {"status": "fit", "n_laps": int(len(clean)), "model": "hinge",
                    "intercept": float(coef[0]), "early_slope_s_per_lap": float(coef[1]),
                    "late_extra_s_per_lap": float(coef[2]), "knot": float(knot),
                    "train_mae_s": mae}
    if best is None or best["early_slope_s_per_lap"] + best["late_extra_s_per_lap"] <= 0:
        return {"status": "below isolation threshold", "n_laps": int(len(clean)), "model": None}
    return best


def predict(model_info: dict, ages) -> np.ndarray:
    if model_info.get("status") != "fit":
        return np.full(len(np.atleast_1d(ages)), np.nan)
    return hinge_value(ages, model_info["intercept"], model_info["early_slope_s_per_lap"],
                      model_info["late_extra_s_per_lap"], model_info["knot"])


def cumulative_degradation(model_info: dict, ages) -> np.ndarray:
    x = np.asarray(ages, dtype=float)
    if model_info.get("status") != "fit":
        return np.full_like(x, np.nan, dtype=float)
    # Delta from first queried age, so cumulative degradation is relative.
    return predict(model_info, x) - predict(model_info, np.array([x.min()]))[0]


def recommend_pit_window(model_info: dict, max_age: int = 60,
                         thresholds=(1.5, 2.5)) -> dict:
    if model_info.get("status") != "fit":
        return {"status": model_info.get("status", "unavailable"), "start_lap": None, "end_lap": None}
    ages = np.arange(1, max_age + 1, dtype=float)
    deg = cumulative_degradation(model_info, ages)
    lo = np.flatnonzero(deg >= thresholds[0])
    hi = np.flatnonzero(deg >= thresholds[1])
    if len(lo) == 0:
        return {"status": "degradation is low", "start_lap": None, "end_lap": None}
    start = int(ages[lo[0]])
    end = int(ages[hi[0]]) if len(hi) else None
    return {"status": "window suggested" if end is not None else "1.5 s threshold reached; 2.5 s not reached",
            "start_lap": start, "end_lap": end}
