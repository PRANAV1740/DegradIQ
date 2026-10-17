"""Stages 2 to 4: physical fuel correction, traffic filtering, track evolution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FUEL_CORRECTION_SECONDS_PER_LAP, TRAFFIC_OUTLIER_SECONDS


def defuel(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each lap to lap-1 fuel load using the specified fixed correction."""
    out = df.copy()
    out["LapNumber"] = pd.to_numeric(out["LapNumber"], errors="coerce")
    out["LapTime_s"] = pd.to_numeric(out["LapTime_s"], errors="coerce")
    out = out.dropna(subset=["LapNumber", "LapTime_s"])
    out["FuelCorrection_s"] = (out["LapNumber"] - 1.0) * FUEL_CORRECTION_SECONDS_PER_LAP
    out["DefuelledTime_s"] = out["LapTime_s"] + out["FuelCorrection_s"]
    return out


def detraffic(df: pd.DataFrame) -> pd.DataFrame:
    """Drop laps more than the specified amount slower than driver/stint median."""
    if df.empty:
        return df.copy()
    out = df.copy()
    keys = [c for c in ("EventName", "Session", "Driver", "Stint") if c in out.columns]
    medians = out.groupby(keys, dropna=False)["DefuelledTime_s"].transform("median")
    out["TrafficLoss_s"] = out["DefuelledTime_s"] - medians
    return out[out["TrafficLoss_s"] <= TRAFFIC_OUTLIER_SECONDS].copy()


def deevolve(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract only a negative field pace trend, capped by each stint's raw range.

    The fitted trend is in seconds per session minute. A negative trend means
    field pace improved as the session progressed. Positive trends are set to
    zero because this implementation only attributes faster pace to rubbering in.
    """
    if df.empty:
        out = df.copy()
        out["TrackCorrection_s"] = []
        out["CorrectedTime_s"] = []
        return out
    out = df.copy()
    out["TrackCorrection_s"] = 0.0
    session_keys = [c for c in ("EventName", "Session") if c in out.columns]
    for _, session in out.groupby(session_keys, dropna=False):
        valid = session.dropna(subset=["SessionMinutes", "DefuelledTime_s"])
        if len(valid) < 3 or valid["SessionMinutes"].nunique() < 2:
            continue
        field = valid.groupby("SessionMinutes", as_index=False)["DefuelledTime_s"].median()
        slope, intercept = np.polyfit(field["SessionMinutes"].to_numpy(float),
                                      field["DefuelledTime_s"].to_numpy(float), 1)
        if not np.isfinite(slope) or slope >= 0:
            continue
        trend = slope * (session["SessionMinutes"] - field["SessionMinutes"].min())
        proposed = np.maximum(0.0, -trend.astype(float)).fillna(0.0)
        stint_keys = [c for c in ("EventName", "Session", "Driver", "Stint") if c in out.columns]
        for _, stint in session.groupby(stint_keys, dropna=False):
            raw_range = float(stint["LapTime_s"].max() - stint["LapTime_s"].min()) if "LapTime_s" in stint.columns else float(stint["DefuelledTime_s"].max() - stint["DefuelledTime_s"].min())
            cap = max(0.0, raw_range)
            idx = stint.index
            out.loc[idx, "TrackCorrection_s"] = np.minimum(proposed.loc[idx], cap)
    out["CorrectedTime_s"] = out["DefuelledTime_s"] + out["TrackCorrection_s"]
    # Track correction is added back to times because the track's pace gain is removed.
    return out
