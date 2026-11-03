"""Independent generalisation validation for the first-stop candidate (offline).

Addresses the open research question from the audit: because ``EventName`` is a
model feature and the same circuits recur across seasons, the chronological
(temporal) holdout only measures *unseen-season, seen-circuit* performance. It
does NOT establish whether the model generalises to a circuit it has never seen.

This harness evaluates two clearly separated regimes on the SAME certified,
label-versioned dataset, reusing the committed training/evaluation logic — no
network access, no re-labelling, no change to the model methodology:

  1. Temporal holdout (unseen season, SEEN circuit): train Season < H, test
     Season == H. This reproduces the development/partial-validation regime.
  2. Leave-one-circuit-out (LOCO) (UNSEEN circuit): for each circuit C, train on
     every other circuit (all seasons) and test on C. The held-out circuit never
     appears in training, so this is a genuine unseen-circuit measurement.

Both regimes report probabilistic metrics (Brier / log-loss / AUC), stop-lap MAE
vs the historical-median baseline, and 10-90 interval **coverage AND width** — the
width being the gap the audit flagged as missing. Metrics are reported per event
and aggregated both micro (pooled rows) and macro (mean over circuits).

Information-regime note (reconciliation): probabilistic metrics are measured over
per-lap rows using in-race features; stop-lap MAE/interval metrics are a *pre-race*
forecast built from starting grid/compound/tyre-age only. The two are never mixed.

Usage (offline)::

    python -m degradiq.scripts.validate_generalization
    python -m degradiq.scripts.validate_generalization --loco --temporal 2024 2025
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

from degradiq.pit_strategy import (  # noqa: E402
    PACKAGE_DIR, FEATURE_COLUMNS, LABEL_DEFINITION_VERSION, _lap_features,
)
from degradiq.evaluation import (  # noqa: E402
    HistoricalMedianBaseline, EmpiricalHazardBaseline, MarginalHazardBaseline,
    compute_calibration_diagnostics,
)
from degradiq.scripts.train_first_stop import (  # noqa: E402
    fit_candidate_first_stop_model, _json_default,
)
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

DEFAULT_DATASET = (PACKAGE_DIR / "cache" / "first_stop_v2_runs" / "run_v2_20261018"
                   / "combined_certified.parquet")
TRACKED_DIR = PACKAGE_DIR / "reports" / "experiments" / "first_stop_v2"
RUNS_DIR = PACKAGE_DIR / "cache" / "first_stop_v2_runs"


def _git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=str(ROOT),
                                stderr=subprocess.DEVNULL)
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _simulate_stop_lap(pipeline, test: pd.DataFrame, median_baseline: HistoricalMedianBaseline
                       ) -> dict[str, Any]:
    """Pre-race stop-lap simulation: MAE, 10-90 interval coverage AND width.

    Mirrors the committed ``evaluate_candidate_chronological`` simulation exactly
    (same pairing discipline), and additionally records the 10-90 interval width.
    """
    cand_err, base_err, hits, widths = [], [], [], []
    per_event: dict[str, dict[str, list]] = {}

    for (event_name, _), group in test.groupby(["EventName", "Driver"], dropna=False):
        group = group.sort_values("LapNumber")
        positives = group[group["StopNextLap"] == 1]
        if positives.empty:
            continue  # right-censored driver excluded from observed stop-lap MAE
        first = group.iloc[0]
        ev = str(first["EventName"])
        compound = str(first.get("Compound", "MEDIUM"))
        actual_lap = int(positives["LapNumber"].iloc[0]) + 1

        base_lap = median_baseline.predict_lap(ev, compound)
        b_err = abs(base_lap - actual_lap)

        driver_state = {
            "driver": str(first["Driver"]), "team": str(first.get("Team", "Unknown")),
            "grid_position": first.get("GridPosition"), "starting_compound": compound,
            "starting_tyre_age": max(1, int(first.get("TyreLife", 1))),
        }
        race_distance = max(5, int(first.get("RaceLaps", group["LapNumber"].max())))
        forecast_rows = [_lap_features(driver_state, lap, race_distance, ev, "pre-race")
                         for lap in range(2, race_distance + 1)]
        if not forecast_rows:
            continue
        f_df = pd.DataFrame(forecast_rows)[FEATURE_COLUMNS]
        gp = pipeline.predict_proba(f_df)[:, 1]
        surv, masses = 1.0, []
        laps = np.arange(2, race_distance + 1)
        for prob in gp:
            masses.append(surv * float(prob))
            surv *= 1.0 - float(prob)
        masses = np.asarray(masses, dtype=float)
        if masses.sum() <= 0:
            continue
        masses /= masses.sum()
        cdf = np.cumsum(masses)

        pred_lap = int(laps[min(np.searchsorted(cdf, 0.50), len(laps) - 1)])
        p10 = int(laps[min(np.searchsorted(cdf, 0.10), len(laps) - 1)])
        p90 = int(laps[min(np.searchsorted(cdf, 0.90), len(laps) - 1)])
        c_err = abs(pred_lap - actual_lap)
        cand_err.append(c_err)
        base_err.append(b_err)
        hits.append(bool(p10 <= actual_lap <= p90))
        widths.append(int(p90 - p10))

        d = per_event.setdefault(ev, {"cand": [], "base": [], "hits": [], "widths": []})
        d["cand"].append(c_err)
        d["base"].append(b_err)
        d["hits"].append(hits[-1])
        d["widths"].append(widths[-1])

    assert len(cand_err) == len(base_err) == len(hits) == len(widths)
    per_event_metrics = {
        ev: {
            "drivers_evaluated": len(v["cand"]),
            "candidate_lap_mae": float(np.mean(v["cand"])) if v["cand"] else None,
            "baseline_median_lap_mae": float(np.mean(v["base"])) if v["base"] else None,
            "interval_coverage_10_90": float(np.mean(v["hits"])) if v["hits"] else None,
            "interval_width_10_90_mean": float(np.mean(v["widths"])) if v["widths"] else None,
        }
        for ev, v in per_event.items()
    }
    return {
        "drivers_with_observed_first_stop": len(cand_err),
        "first_stop_lap_mae": float(np.mean(cand_err)) if cand_err else None,
        "baseline_median_lap_mae": float(np.mean(base_err)) if base_err else None,
        "interval_coverage_10_90": float(np.mean(hits)) if hits else None,
        "interval_width_10_90_mean": float(np.mean(widths)) if widths else None,
        "interval_width_10_90_median": float(np.median(widths)) if widths else None,
        "per_event_metrics": per_event_metrics,
        "_errors": cand_err,  # internal, for macro/micro aggregation
    }


def _probabilistic(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pc = np.clip(p, 1e-8, 1 - 1e-8)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, pc, labels=[0, 1])),
        "roc_auc": auc,
        "n_rows": int(len(y)),
        "observed_stop_rate": float(y.mean()),
    }


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, tag: str) -> dict[str, Any]:
    """Fit the committed candidate + baselines on ``train``; evaluate on ``test``."""
    bundle = fit_candidate_first_stop_model(train, architecture="calibrated_hgb")
    pipeline = bundle["pipeline"]
    median_baseline = HistoricalMedianBaseline().fit(train)
    empirical = EmpiricalHazardBaseline().fit(train)
    marginal = MarginalHazardBaseline().fit(train)

    y = test["StopNextLap"].astype(int).to_numpy()
    p = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    prob = _probabilistic(y, p)
    calib = compute_calibration_diagnostics(y, p)
    sim = _simulate_stop_lap(pipeline, test, median_baseline)

    return {
        "tag": tag,
        "train_rows": int(len(train)), "holdout_rows": int(len(test)),
        "train_circuits": int(train["EventName"].nunique()),
        "candidate": {**prob, **{k: sim[k] for k in (
            "first_stop_lap_mae", "interval_coverage_10_90",
            "interval_width_10_90_mean", "interval_width_10_90_median",
            "drivers_with_observed_first_stop")},
            "expected_calibration_error": calib["expected_calibration_error"]},
        "baseline_historical_median_mae": sim["baseline_median_lap_mae"],
        "baseline_empirical_hazard": empirical.evaluate(test),
        "baseline_marginal_hazard": marginal.evaluate(test),
        "per_event_metrics": sim["per_event_metrics"],
        "_test_pred": (y, p),         # internal
        "_errors": sim["_errors"],    # internal
    }


def run_temporal(rows: pd.DataFrame, holdout_years: list[int]) -> list[dict[str, Any]]:
    out = []
    for h in holdout_years:
        train = rows[rows["Season"] < h]
        test = rows[rows["Season"] == h]
        if train.empty or test.empty:
            out.append({"tag": f"temporal_holdout_{h}", "status": "skipped_no_rows"})
            continue
        print(f"[temporal {h}] train={len(train)} test={len(test)} ...")
        rep = evaluate_split(train, test, f"temporal_holdout_{h}")
        rep.pop("_test_pred", None)
        rep.pop("_errors", None)
        out.append(rep)
    return out


def run_loco(rows: pd.DataFrame) -> dict[str, Any]:
    circuits = sorted(rows["EventName"].dropna().astype(str).unique())
    per_circuit = {}
    pooled_y, pooled_p, pooled_errors = [], [], []
    macro_keys = ("brier_score", "log_loss", "roc_auc", "first_stop_lap_mae",
                  "interval_coverage_10_90", "interval_width_10_90_mean")
    macro: dict[str, list[float]] = {k: [] for k in macro_keys}

    for i, c in enumerate(circuits, 1):
        train = rows[rows["EventName"].astype(str) != c]
        test = rows[rows["EventName"].astype(str) == c]
        if test.empty or train.empty:
            continue
        print(f"[LOCO {i}/{len(circuits)}] held-out circuit='{c}' "
              f"(train={len(train)}, test={len(test)}) ...")
        rep = evaluate_split(train, test, f"loco_{c}")
        y, p = rep.pop("_test_pred")
        errs = rep.pop("_errors")
        pooled_y.append(y)
        pooled_p.append(p)
        pooled_errors.extend(errs)
        cm = rep["candidate"]
        per_circuit[c] = {
            "holdout_rows": rep["holdout_rows"],
            "brier_score": cm["brier_score"], "log_loss": cm["log_loss"],
            "roc_auc": cm["roc_auc"], "first_stop_lap_mae": cm["first_stop_lap_mae"],
            "baseline_historical_median_mae": rep["baseline_historical_median_mae"],
            "interval_coverage_10_90": cm["interval_coverage_10_90"],
            "interval_width_10_90_mean": cm["interval_width_10_90_mean"],
            "drivers_with_observed_first_stop": cm["drivers_with_observed_first_stop"],
        }
        for k in macro_keys:
            v = per_circuit[c][k]
            if v is not None:
                macro[k].append(v)

    y_all = np.concatenate(pooled_y) if pooled_y else np.array([])
    p_all = np.concatenate(pooled_p) if pooled_p else np.array([])
    micro = _probabilistic(y_all, p_all) if len(y_all) else {}
    micro_mae = float(np.mean(pooled_errors)) if pooled_errors else None

    # Count where the candidate's pre-race timing MAE matches/beats the historical-
    # median baseline on the UNSEEN circuit. Reported so this headline claim is
    # backed by a tracked number, not only prose.
    evaluable = [v for v in per_circuit.values()
                 if v["first_stop_lap_mae"] is not None
                 and v["baseline_historical_median_mae"] is not None]
    le = sum(1 for v in evaluable
             if v["first_stop_lap_mae"] <= v["baseline_historical_median_mae"])
    lt = sum(1 for v in evaluable
             if v["first_stop_lap_mae"] < v["baseline_historical_median_mae"])
    return {
        "n_circuits": len(per_circuit),
        "micro_pooled": {**micro, "first_stop_lap_mae": micro_mae,
                         "drivers_with_observed_first_stop": len(pooled_errors)},
        "macro_mean_over_circuits": {k: (float(np.mean(v)) if v else None)
                                     for k, v in macro.items()},
        "timing_mae_vs_baseline": {
            "circuits_evaluated": len(evaluable),
            "candidate_le_baseline": le,   # matches or beats (ties included)
            "candidate_lt_baseline": lt,   # strictly beats
        },
        "per_circuit": per_circuit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--temporal", type=int, nargs="*", default=[2024, 2025],
                        help="Holdout years for the temporal (seen-circuit) regime.")
    parser.add_argument("--loco", action="store_true", help="Run leave-one-circuit-out only.")
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Certified dataset not found: {args.dataset}. This script is offline and does "
            "NOT download data; run the v2 pipeline/reproduction first to produce it.")
    rows = pd.read_parquet(args.dataset)
    versions = set(pd.Series(rows.get("LabelDefVersion")).dropna().unique())
    if versions != {LABEL_DEFINITION_VERSION}:
        raise SystemExit(f"Dataset label version {sorted(versions)} != {LABEL_DEFINITION_VERSION}.")

    libs = {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "libraries": libs,
        "dataset_rows": int(len(rows)),
        "dataset_positives": int(rows["StopNextLap"].sum()),
        "regime_note": (
            "temporal_holdout = unseen season, SEEN circuit (EventName recurs). "
            "leave_one_circuit_out = UNSEEN circuit. Probabilistic metrics use in-race "
            "per-lap features; stop-lap MAE/interval metrics are a pre-race forecast."),
    }

    if not args.loco:
        report["temporal_holdout"] = run_temporal(rows, args.temporal)
    if not args.temporal_only:
        print("\n=== Leave-one-circuit-out (unseen circuit) ===")
        report["leave_one_circuit_out"] = run_loco(rows)

    out_dir = args.out or (RUNS_DIR / f"validation_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generalization_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    # Tracked summary (small): headline comparison of the two regimes.
    summary = {k: report[k] for k in ("generated_at_utc", "git_revision",
               "label_definition_version", "libraries", "dataset_rows",
               "dataset_positives", "regime_note")}
    if "temporal_holdout" in report:
        summary["temporal_holdout"] = [
            {"tag": r.get("tag"), "candidate": r.get("candidate"),
             "baseline_historical_median_mae": r.get("baseline_historical_median_mae")}
            for r in report["temporal_holdout"] if "candidate" in r]
    if "leave_one_circuit_out" in report:
        loco = report["leave_one_circuit_out"]
        summary["leave_one_circuit_out"] = {
            "n_circuits": loco["n_circuits"],
            "micro_pooled": loco["micro_pooled"],
            "macro_mean_over_circuits": loco["macro_mean_over_circuits"],
            "timing_mae_vs_baseline": loco["timing_mae_vs_baseline"],
            "per_circuit": loco["per_circuit"],
        }
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
    (TRACKED_DIR / "generalization_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    print("\n" + "=" * 70)
    if "temporal_holdout" in report:
        for r in report["temporal_holdout"]:
            if "candidate" in r:
                c = r["candidate"]
                print(f"  {r['tag']:22s} MAE={c['first_stop_lap_mae']:.3f} "
                      f"AUC={c['roc_auc']:.4f} cov={c['interval_coverage_10_90']:.3f} "
                      f"width={c['interval_width_10_90_mean']:.2f}")
    if "leave_one_circuit_out" in report:
        m = report["leave_one_circuit_out"]["micro_pooled"]
        mm = report["leave_one_circuit_out"]["macro_mean_over_circuits"]
        print(f"  LOCO micro-pooled      MAE={m['first_stop_lap_mae']:.3f} "
              f"AUC={m.get('roc_auc'):.4f}")
        print(f"  LOCO macro-over-circuit MAE={mm['first_stop_lap_mae']:.3f} "
              f"AUC={mm['roc_auc']:.4f} cov={mm['interval_coverage_10_90']:.3f} "
              f"width={mm['interval_width_10_90_mean']:.2f}")
    print(f"  full report: {out_dir / 'generalization_report.json'}")
    print(f"  tracked summary: {TRACKED_DIR / 'generalization_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
