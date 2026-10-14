"""Evaluation baselines and metrics for first-pit-stop hazard models.

Implements Phase 5 and Phase 6 requirements:
1. Historical Median Stop-Lap Baseline (fitted strictly on training data).
2. Empirical Circuit/Compound Hazard Baseline (fitted strictly on training data).
3. Marginal Hazard Baseline (constant empirical hazard on training data).
4. Side-by-side probabilistic, calibration, and lap-error evaluation on holdout data.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class HistoricalMedianBaseline:
    """Historical median stop-lap baseline using training data only.

    Hierarchical lookup:
    1. (EventName, StartingCompound)
    2. (StartingCompound)
    3. Global training median
    """

    def __init__(self):
        self.event_compound_medians: dict[tuple[str, str], float] = {}
        self.compound_medians: dict[str, float] = {}
        self.global_median: float = 20.0
        self.fitted: bool = False

    def fit(self, train_rows: pd.DataFrame) -> "HistoricalMedianBaseline":
        records = []
        for (_, _), group in train_rows.groupby(["EventName", "Driver"], dropna=False):
            group = group.sort_values("LapNumber")
            positives = group[group["StopNextLap"] == 1]
            if positives.empty:
                continue
            first_row = group.iloc[0]
            actual_stop_lap = float(positives["LapNumber"].iloc[0]) + 1.0
            event = str(first_row.get("EventName", "Unknown"))
            compound = str(first_row.get("Compound", "MEDIUM")).upper()
            records.append({
                "EventName": event,
                "Compound": compound,
                "StopLap": actual_stop_lap,
            })

        if not records:
            self.fitted = True
            return self

        obs_df = pd.DataFrame(records)
        self.global_median = float(obs_df["StopLap"].median())

        for compound, sub in obs_df.groupby("Compound"):
            self.compound_medians[str(compound)] = float(sub["StopLap"].median())

        for (event, compound), sub in obs_df.groupby(["EventName", "Compound"]):
            self.event_compound_medians[(str(event), str(compound))] = float(sub["StopLap"].median())

        self.fitted = True
        return self

    def predict_lap(self, event_name: str, compound: str) -> float:
        key = (event_name, compound.upper())
        if key in self.event_compound_medians:
            return self.event_compound_medians[key]
        comp_key = compound.upper()
        if comp_key in self.compound_medians:
            return self.compound_medians[comp_key]
        return self.global_median

    def evaluate(self, holdout_rows: pd.DataFrame) -> dict[str, Any]:
        errors = []
        for (_, _), group in holdout_rows.groupby(["EventName", "Driver"], dropna=False):
            group = group.sort_values("LapNumber")
            positives = group[group["StopNextLap"] == 1]
            if positives.empty:
                continue  # Censored observation excluded from observed stop-lap MAE
            first = group.iloc[0]
            event = str(first.get("EventName", "Unknown"))
            compound = str(first.get("Compound", "MEDIUM"))
            actual_lap = float(positives["LapNumber"].iloc[0]) + 1.0
            pred_lap = self.predict_lap(event, compound)
            errors.append(abs(pred_lap - actual_lap))

        return {
            "baseline_type": "historical_median",
            "drivers_evaluated": len(errors),
            "lap_mae": float(np.mean(errors)) if errors else None,
            "lap_median_error": float(np.median(errors)) if errors else None,
        }


class EmpiricalHazardBaseline:
    """Empirical circuit/compound hazard baseline estimated from training data only."""

    def __init__(self, min_risk_set: int = 5):
        self.min_risk_set = min_risk_set
        self.event_compound_lap_hazard: dict[tuple[str, str, int], float] = {}
        self.compound_lap_hazard: dict[tuple[str, int], float] = {}
        self.lap_hazard: dict[int, float] = {}
        self.marginal_rate: float = 0.05
        self.fitted: bool = False

    def fit(self, train_rows: pd.DataFrame) -> "EmpiricalHazardBaseline":
        df = train_rows.copy()
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(1).astype(int)
        df["StopNextLap"] = pd.to_numeric(df["StopNextLap"], errors="coerce").fillna(0).astype(int)
        df["Compound"] = df["Compound"].astype(str).str.upper()
        df["EventName"] = df["EventName"].astype(str)

        total_rows = len(df)
        total_stops = int(df["StopNextLap"].sum())
        self.marginal_rate = float(total_stops / max(1, total_rows))

        # Overall lap hazard
        for lap, sub in df.groupby("LapNumber"):
            n_risk = len(sub)
            n_events = int(sub["StopNextLap"].sum())
            lap_int = int(str(lap).split(".")[0])
            self.lap_hazard[lap_int] = float(n_events / max(1, n_risk))

        # Compound x lap hazard
        for (compound, lap), sub in df.groupby(["Compound", "LapNumber"]):
            n_risk = len(sub)
            n_events = int(sub["StopNextLap"].sum())
            if n_risk >= self.min_risk_set:
                self.compound_lap_hazard[(str(compound), int(lap))] = float(n_events / n_risk)

        # Event x compound x lap hazard
        for (event, compound, lap), sub in df.groupby(["EventName", "Compound", "LapNumber"]):
            n_risk = len(sub)
            n_events = int(sub["StopNextLap"].sum())
            if n_risk >= self.min_risk_set:
                self.event_compound_lap_hazard[(str(event), str(compound), int(lap))] = float(n_events / n_risk)

        self.fitted = True
        return self

    def predict_hazard_row(self, event: str, compound: str, lap: int) -> float:
        key3 = (event, compound.upper(), lap)
        if key3 in self.event_compound_lap_hazard:
            return self.event_compound_lap_hazard[key3]
        key2 = (compound.upper(), lap)
        if key2 in self.compound_lap_hazard:
            return self.compound_lap_hazard[key2]
        if lap in self.lap_hazard:
            return self.lap_hazard[lap]
        return self.marginal_rate

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        hazards = []
        for _, row in rows.iterrows():
            event = str(row.get("EventName", "Unknown"))
            compound = str(row.get("Compound", "MEDIUM"))
            lap = int(pd.to_numeric(row.get("LapNumber", 1), errors="coerce") or 1)
            h = self.predict_hazard_row(event, compound, lap)
            hazards.append(h)
        return np.asarray(hazards, dtype=float)

    def evaluate(self, holdout_rows: pd.DataFrame) -> dict[str, Any]:
        y_test = holdout_rows["StopNextLap"].astype(int).to_numpy()
        p = self.predict_proba(holdout_rows)
        p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
        return {
            "baseline_type": "empirical_circuit_compound_hazard",
            "brier_score": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, p_clipped, labels=[0, 1])),
            "mean_predicted_hazard": float(np.mean(p)),
        }


class MarginalHazardBaseline:
    """Marginal constant hazard baseline on training data."""

    def __init__(self):
        self.marginal_rate: float = 0.05
        self.fitted: bool = False

    def fit(self, train_rows: pd.DataFrame) -> "MarginalHazardBaseline":
        total_rows = len(train_rows)
        total_stops = train_rows["StopNextLap"].astype(int).sum()
        self.marginal_rate = float(total_stops / max(1, total_rows))
        self.fitted = True
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.marginal_rate, dtype=float)

    def evaluate(self, holdout_rows: pd.DataFrame) -> dict[str, Any]:
        y_test = holdout_rows["StopNextLap"].astype(int).to_numpy()
        p = self.predict_proba(holdout_rows)
        p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
        return {
            "baseline_type": "marginal_constant_hazard",
            "marginal_rate": self.marginal_rate,
            "brier_score": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, p_clipped, labels=[0, 1])),
        }


def compute_calibration_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> dict[str, Any]:
    """Calculate calibration error across quantile bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_diagnostics = []
    total_abs_error = 0.0
    valid_samples = 0

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count == 0:
            continue
        actual_rate = float(np.mean(y_true[mask]))
        pred_rate = float(np.mean(y_prob[mask]))
        error = abs(actual_rate - pred_rate)
        total_abs_error += error * count
        valid_samples += count
        bin_diagnostics.append({
            "bin_range": [round(float(bins[i]), 3), round(float(bins[i + 1]), 3)],
            "count": count,
            "observed_frequency": round(actual_rate, 4),
            "predicted_probability": round(pred_rate, 4),
            "abs_error": round(error, 4),
        })

    ece = float(total_abs_error / max(1, valid_samples))
    return {
        "expected_calibration_error": round(ece, 4),
        "binned_calibration": bin_diagnostics,
    }
