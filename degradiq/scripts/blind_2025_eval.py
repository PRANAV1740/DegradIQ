"""Blind-ish out-of-sample evaluation on the full 2025 season.

The 13 events that were unavailable in the v2 run were never used in training or in
any development/validation decision. Evaluating the frozen 2022-2024 model on them is
therefore a genuine out-of-sample test. This script:

  1. Recovers the missing 2025 events via FastF1 into an ISOLATED checkpoint
     directory (never the legacy dir; the original run is untouched). Already-cached
     sessions load from the FastF1 cache; the rest download under the rate limiter.
  2. Builds the full 2025 holdout and combines it with the certified 2022-2024
     training rows from the canonical run.
  3. Fits the frozen candidate on 2022-2024 (identical to the promoted model) and the
     baselines, then evaluates on: full 2025, the 13 recovered events (true blind),
     and the 11 previously examined events — reported separately.

Needs network for the 13 missing events. Run:

    python -m degradiq.scripts.blind_2025_eval
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

from degradiq.pit_strategy import PACKAGE_DIR, FEATURE_COLUMNS, LABEL_DEFINITION_VERSION
from degradiq.rate_limiter import RateLimiter, CheckpointManager
from degradiq.evaluation import HistoricalMedianBaseline
from degradiq.scripts.train_first_stop import (
    collect_rows_with_checkpoints, fit_candidate_first_stop_model, _json_default,
)
from degradiq.scripts.validate_generalization import _simulate_stop_lap, _probabilistic

CERTIFIED = (PACKAGE_DIR / "cache" / "first_stop_v2_runs" / "run_v2_20261018"
             / "combined_certified.parquet")
RUNS_DIR = PACKAGE_DIR / "cache" / "first_stop_v2_runs"
TRACKED_DIR = PACKAGE_DIR / "reports" / "experiments" / "first_stop_v2"

# Events examined during v2 development (loaded in the original run) vs the 13 that
# were unavailable then and so never influenced development.
EXAMINED_11 = {
    "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix",
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix",
    "Canadian Grand Prix", "Austrian Grand Prix",
}


def _evaluate(pipeline, median_baseline, test: pd.DataFrame, tag: str) -> dict:
    if test.empty:
        return {"tag": tag, "status": "no_rows"}
    y = test["StopNextLap"].astype(int).to_numpy()
    p = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    sim = _simulate_stop_lap(pipeline, test, median_baseline)
    return {
        "tag": tag,
        "events": int(test["EventName"].nunique()),
        "holdout_rows": int(len(test)),
        "holdout_positive_rows": int(y.sum()),
        "probabilistic": _probabilistic(y, p),
        "first_stop_lap_mae": sim["first_stop_lap_mae"],
        "baseline_historical_median_mae": sim["baseline_median_lap_mae"],
        "interval_coverage_10_90": sim["interval_coverage_10_90"],
        "interval_width_10_90_mean": sim["interval_width_10_90_mean"],
        "drivers_with_observed_first_stop": sim["drivers_with_observed_first_stop"],
    }


def main() -> None:
    if not CERTIFIED.exists():
        raise SystemExit(f"Certified dataset missing: {CERTIFIED}")
    certified = pd.read_parquet(CERTIFIED)
    train = certified[certified["Season"] < 2025].copy()
    if "LabelDefVersion" not in train.columns:
        train["LabelDefVersion"] = LABEL_DEFINITION_VERSION

    run_id = f"blind_2025_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RUNS_DIR / run_id
    ckpt_dir = out_dir / "checkpoints_2025"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[recover] downloading/building full 2025 into isolated dir: {ckpt_dir}")
    mgr = CheckpointManager(ckpt_dir)
    limiter = RateLimiter(min_interval_s=9.0)
    rows_2025, coverage_2025 = collect_rows_with_checkpoints(
        2025, 2025, mgr, limiter, offline=False, mode="train")
    coverage_2025.to_csv(out_dir / "coverage_2025.csv", index=False)

    loaded = coverage_2025[coverage_2025["status"] == "loaded"]
    print(f"[recover] 2025 events loaded: {len(loaded)} / {len(coverage_2025)}")
    if rows_2025.empty:
        raise SystemExit("No 2025 rows recovered; aborting.")
    if "LabelDefVersion" not in rows_2025.columns:
        rows_2025["LabelDefVersion"] = LABEL_DEFINITION_VERSION

    # Frozen model on 2022-2024 (identical to the promoted candidate).
    print("[fit] frozen candidate on 2022-2024 ...")
    bundle = fit_candidate_first_stop_model(train, architecture="calibrated_hgb")
    pipeline = bundle["pipeline"]
    median_baseline = HistoricalMedianBaseline().fit(train)

    recovered_events = sorted(set(rows_2025["EventName"].unique()) - EXAMINED_11)
    blind = rows_2025[rows_2025["EventName"].isin(recovered_events)]
    examined = rows_2025[rows_2025["EventName"].isin(EXAMINED_11)]

    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "frozen_training_seasons": [2022, 2023, 2024],
        "train_rows": int(len(train)),
        "events_loaded_2025": int(len(loaded)),
        "events_scheduled_2025": int(len(coverage_2025)),
        "recovered_events": recovered_events,
        "honesty_note": (
            "blind_13_recovered = events unavailable during v2 development, so they never "
            "influenced training or model selection: a genuine out-of-sample test of the "
            "frozen 2022-2024 model. examined_11 = events seen during development. "
            "full_2025 pools both. Probabilistic metrics use in-race per-lap features; "
            "stop-lap MAE/interval metrics are a pre-race forecast."),
        "results": {
            "full_2025": _evaluate(pipeline, median_baseline, rows_2025, "full_2025"),
            "blind_13_recovered": _evaluate(pipeline, median_baseline, blind, "blind_13_recovered"),
            "examined_11": _evaluate(pipeline, median_baseline, examined, "examined_11"),
        },
    }
    for r in report["results"].values():
        r.pop("_errors", None)

    (out_dir / "blind_2025_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
    (TRACKED_DIR / "blind_2025_summary.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    print("\n" + "=" * 70)
    for key, r in report["results"].items():
        if r.get("status") == "no_rows":
            print(f"  {key:20s} no rows")
            continue
        pr = r["probabilistic"]
        print(f"  {key:20s} events={r['events']:2d} MAE={r['first_stop_lap_mae']} "
              f"base={r['baseline_historical_median_mae']} AUC={pr['roc_auc']} "
              f"Brier={pr['brier_score']:.4f} cov={r['interval_coverage_10_90']} "
              f"width={r['interval_width_10_90_mean']}")
    print(f"  tracked summary: {TRACKED_DIR / 'blind_2025_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
