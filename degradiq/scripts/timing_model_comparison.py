"""Phase III: compare first-stop TIMING models under LOCO + temporal evaluation.

Motivation: the served calibrated-HGB *hazard* model generalises for classification
but its stop-lap timing beats the historical-median baseline on only 5/25 unseen
circuits, and its 10-90 interval is ~22 laps wide. The worst circuits (Hungary,
Dutch, Las Vegas, China) have tight, predictable stop windows that a diffuse
survival-derived median handles poorly.

This script fairly compares the hazard model against a timing-specific candidate — a
**direct quantile stop-lap regressor** (gradient boosting with pinball loss at the
10th/50th/90th percentiles) — plus an EventName ablation, under the SAME
leave-one-circuit-out (unseen circuit) and temporal (unseen season) splits, with the
SAME historical-median baseline. It reports MAE, 10-90 interval coverage AND width.
No model is promoted here; this produces the evidence for a selection decision.

Offline. Reads the certified dataset; reuses the stored HGB LOCO numbers from
generalization_summary.json so the hazard model is not re-fit.

    python -m degradiq.scripts.timing_model_comparison
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from degradiq.pit_strategy import PACKAGE_DIR, LABEL_DEFINITION_VERSION

CERTIFIED = (PACKAGE_DIR / "cache" / "first_stop_v2_runs" / "run_v2_20261018"
             / "combined_certified.parquet")
TRACKED_DIR = PACKAGE_DIR / "reports" / "experiments" / "first_stop_v2"
RUNS_DIR = PACKAGE_DIR / "cache" / "first_stop_v2_runs"

NUM = ["RaceLaps", "GridPosition", "StartTyreLife"]
CAT_FULL = ["Compound", "Team", "EventName"]
CAT_NOEVENT = ["Compound", "Team"]
QUANTILES = (0.1, 0.5, 0.9)


def driver_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, driver) with observed first stop: pre-race features + stop lap."""
    recs = []
    # Key on season too: the same driver+circuit recurs each year as a distinct race.
    for (ev, drv, season), g in rows.groupby(["EventName", "Driver", "Season"], dropna=False):
        g = g.sort_values("LapNumber")
        pos = g[g["StopNextLap"] == 1]
        if pos.empty:
            continue
        first = g.iloc[0]
        grid = first.get("GridPosition")
        recs.append({
            "EventName": str(ev), "Driver": str(drv),
            "Team": str(first.get("Team", "Unknown")),
            "Compound": str(first.get("Compound", "MEDIUM")).upper(),
            "RaceLaps": float(first.get("RaceLaps", g["LapNumber"].max())),
            "GridPosition": float(grid) if pd.notna(grid) else np.nan,
            "StartTyreLife": float(first.get("TyreLife", 1)),
            "Season": int(first.get("Season", 0)),
            "StopLap": int(pos["LapNumber"].iloc[0]) + 1,
        })
    return pd.DataFrame(recs)


def _median_baseline(train: pd.DataFrame):
    glob = float(train["StopLap"].median())
    comp = train.groupby("Compound")["StopLap"].median().to_dict()
    ec = train.groupby(["EventName", "Compound"])["StopLap"].median().to_dict()

    def predict(ev: str, compound: str) -> float:
        return float(ec.get((ev, compound), comp.get(compound, glob)))
    return predict


def _fit_quantiles(train: pd.DataFrame, cats: list[str]):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    models = {}
    X, y = train[NUM + cats], train["StopLap"].to_numpy()
    for q in QUANTILES:
        m = Pipeline([("pre", pre), ("gbr", GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=200, max_depth=3,
            learning_rate=0.05, random_state=42))])
        m.fit(X, y)
        models[q] = m
    return models, cats


def _predict_quantiles(models, cats, test: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([models[q].predict(test[NUM + cats]) for q in QUANTILES])
    preds.sort(axis=1)  # enforce q10 <= q50 <= q90
    return preds


def _metrics(q10, q50, q90, actual, base) -> dict:
    actual = np.asarray(actual, float)
    mae = float(np.mean(np.abs(q50 - actual)))
    base_mae = float(np.mean(np.abs(np.asarray(base, float) - actual)))
    cover = float(np.mean((q10 <= actual) & (actual <= q90)))
    width = float(np.mean(q90 - q10))
    return {"n": int(len(actual)), "mae": mae, "baseline_mae": base_mae,
            "interval_coverage_10_90": cover, "interval_width_10_90_mean": width}


def _eval_quantile(table: pd.DataFrame, cats: list[str], mode: str) -> dict:
    """mode='loco' or 'temporal_2024'. Returns micro + macro + per-circuit."""
    per, macro = {}, {"mae": [], "baseline_mae": [], "interval_coverage_10_90": [],
                      "interval_width_10_90_mean": []}
    all_q50, all_q10, all_q90, all_actual, all_base = [], [], [], [], []

    if mode == "loco":
        splits = [(c, table["EventName"] != c, table["EventName"] == c)
                  for c in sorted(table["EventName"].unique())]
    else:  # temporal: train seasons < 2024, test == 2024
        splits = [("2024", table["Season"] < 2024, table["Season"] == 2024)]

    for name, trmask, temask in splits:
        train, test = table[trmask], table[temask]
        if len(train) < 50 or test.empty:
            continue
        models, _ = _fit_quantiles(train, cats)
        preds = _predict_quantiles(models, cats, test)
        q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]
        base_fn = _median_baseline(train)
        base = [base_fn(e, c) for e, c in zip(test["EventName"], test["Compound"])]
        m = _metrics(q10, q50, q90, test["StopLap"], base)
        per[name] = m
        for k in macro:
            macro[k].append(m[k])
        all_q50.append(q50); all_q10.append(q10); all_q90.append(q90)
        all_actual.append(test["StopLap"].to_numpy()); all_base.append(np.asarray(base))

    if mode == "loco":
        micro = _metrics(np.concatenate(all_q10), np.concatenate(all_q50),
                         np.concatenate(all_q90), np.concatenate(all_actual),
                         np.concatenate(all_base))
        beats = sum(1 for v in per.values() if v["mae"] <= v["baseline_mae"])
        return {"micro_pooled": micro,
                "macro_mean": {k: float(np.mean(v)) for k, v in macro.items()},
                "circuits_evaluated": len(per),
                "circuits_candidate_le_baseline": beats,
                "per_circuit": per}
    return {"result": per.get("2024")}


def main() -> None:
    if not CERTIFIED.exists():
        raise SystemExit(f"Certified dataset missing: {CERTIFIED}")
    rows = pd.read_parquet(CERTIFIED)
    table = driver_table(rows)

    print(f"driver-stop table: {len(table)} driver-stops across "
          f"{table['EventName'].nunique()} circuits")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "note": ("Timing-only comparison. quantile_gbr = direct gradient-boosted quantile "
                 "regression of the first-stop LAP (pinball loss at 0.1/0.5/0.9). It is "
                 "conditional on an observed stop (censoring ignored), same as the hazard "
                 "model's stop-lap MAE. Baseline = historical-median stop lap. "
                 "hazard_hgb_loco is read from generalization_summary.json (not re-fit)."),
        "quantile_gbr": {
            "loco": _eval_quantile(table, CAT_FULL, "loco"),
            "temporal_2024": _eval_quantile(table, CAT_FULL, "temporal_2024"),
        },
        "quantile_gbr_no_eventname_ablation": {
            "loco": _eval_quantile(table, CAT_NOEVENT, "loco"),
        },
    }

    # Pull the stored HGB hazard LOCO timing for a side-by-side.
    gen = json.loads((TRACKED_DIR / "generalization_summary.json").read_text(encoding="utf-8"))
    loco = gen["leave_one_circuit_out"]
    report["hazard_hgb_reference"] = {
        "loco_macro_mae": loco["macro_mean_over_circuits"]["first_stop_lap_mae"],
        "loco_macro_coverage": loco["macro_mean_over_circuits"]["interval_coverage_10_90"],
        "loco_macro_width": loco["macro_mean_over_circuits"]["interval_width_10_90_mean"],
        "loco_circuits_candidate_le_baseline": loco["timing_mae_vs_baseline"]["candidate_le_baseline"],
        "loco_circuits_evaluated": loco["timing_mae_vs_baseline"]["circuits_evaluated"],
    }

    out_dir = RUNS_DIR / f"timing_cmp_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
    (TRACKED_DIR / "timing_comparison_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    qg = report["quantile_gbr"]["loco"]
    qa = report["quantile_gbr_no_eventname_ablation"]["loco"]
    hz = report["hazard_hgb_reference"]
    print("\n" + "=" * 72)
    print("  UNSEEN-CIRCUIT (LOCO) TIMING")
    print(f"  hazard_hgb (current)   macroMAE={hz['loco_macro_mae']:.2f}  "
          f"cov={hz['loco_macro_coverage']:.2f}  width={hz['loco_macro_width']:.1f}  "
          f"beats_base={hz['loco_circuits_candidate_le_baseline']}/{hz['loco_circuits_evaluated']}")
    print(f"  quantile_gbr           macroMAE={qg['macro_mean']['mae']:.2f}  "
          f"cov={qg['macro_mean']['interval_coverage_10_90']:.2f}  "
          f"width={qg['macro_mean']['interval_width_10_90_mean']:.1f}  "
          f"beats_base={qg['circuits_candidate_le_baseline']}/{qg['circuits_evaluated']}")
    print(f"  quantile_gbr (no Event) macroMAE={qa['macro_mean']['mae']:.2f}  "
          f"cov={qa['macro_mean']['interval_coverage_10_90']:.2f}  "
          f"width={qa['macro_mean']['interval_width_10_90_mean']:.1f}  "
          f"beats_base={qa['circuits_candidate_le_baseline']}/{qa['circuits_evaluated']}")
    t = report["quantile_gbr"]["temporal_2024"]["result"]
    print(f"  [temporal 2024] quantile_gbr MAE={t['mae']:.2f} base={t['baseline_mae']:.2f} "
          f"cov={t['interval_coverage_10_90']:.2f} width={t['interval_width_10_90_mean']:.1f}")
    print(f"  tracked: {TRACKED_DIR / 'timing_comparison_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
