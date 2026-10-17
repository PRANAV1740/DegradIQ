"""Evaluate frozen candidate first-stop model on 2025 season holdout data.

Trains frozen candidate and baselines strictly on 2022-2024 (22 events in 2022, 22 in 2023, 24 in 2024; 68 events total).
Evaluates on all recovered 2025 holdout events.
Generates full 24-event 2025 coverage report and frozen holdout backtest report.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import fastf1
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.rate_limiter import CheckpointManager
from degradiq.pit_strategy import (
    PACKAGE_DIR, FEATURE_COLUMNS, _lap_features,
)
from degradiq.evaluation import (
    HistoricalMedianBaseline,
    EmpiricalHazardBaseline,
    MarginalHazardBaseline,
    compute_calibration_diagnostics,
)
from degradiq.scripts.train_first_stop import (
    collect_rows_with_checkpoints,
    fit_candidate_first_stop_model,
    save_candidate_model,
    _json_default,
)

logger = logging.getLogger("degradiq.eval_2025")
REPORT_DIR = PACKAGE_DIR / "reports"
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"


def run_2025_holdout_evaluation():
    mgr = CheckpointManager(PACKAGE_DIR / "cache" / "hazard_checkpoints")
    
    # 1. Collect training data: Seasons 2022-2024
    print("Collecting training rows (2022-2024)...")
    train_rows, train_cov = collect_rows_with_checkpoints(2022, 2024, mgr, None, offline=True)
    print(f"Training set: {len(train_rows)} rows across seasons {train_rows['Season'].unique()}")

    # 2. Collect 2025 holdout schedule and checkpoints
    print("\nAuditing 2025 schedule and checkpoints...")
    schedule = fastf1.get_event_schedule(2025, include_testing=False)
    cov_records = []
    holdout_dfs = []

    for _, event in schedule.iterrows():
        name = str(event.get("EventName", "Unknown"))
        round_id = event.get("RoundNumber", name)
        if pd.isna(round_id) or round_id == 0:
            continue
        round_id = int(round_id)
        
        # Check if valid checkpoint exists
        if mgr.has_valid_checkpoint(2025, round_id, FEATURE_COLUMNS, event_name=name):
            df_cp, meta = mgr.load_checkpoint(2025, round_id, event_name=name)
            status = meta.get("status", "loaded")
            row_count = meta.get("row_count", len(df_cp))
            pos_count = meta.get("positive_count", 0)
            if not df_cp.empty:
                df_cp["Season"] = 2025
                holdout_dfs.append(df_cp)
            cov_records.append({
                "year": 2025,
                "round": round_id,
                "event": name,
                "status": status,
                "rows": row_count,
                "positive_rows": pos_count,
                "error": meta.get("error", ""),
            })
        else:
            # Checkpoint not yet recovered or unavailable
            cov_records.append({
                "year": 2025,
                "round": round_id,
                "event": name,
                "status": "not ingested locally (season complete)",
                "rows": 0,
                "positive_rows": 0,
                "error": "No local checkpoint present; race not ingested in this workspace",
            })

    cov_2025_df = pd.DataFrame(cov_records)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cov_path = REPORT_DIR / "candidate_2025_coverage.csv"
    cov_2025_df.to_csv(cov_path, index=False)
    print(f"Saved 2025 coverage report: {cov_path}")
    print(cov_2025_df[["round", "event", "status", "rows", "positive_rows"]])

    holdout_rows = pd.concat(holdout_dfs, ignore_index=True) if holdout_dfs else pd.DataFrame()
    total_scheduled = len(cov_2025_df)
    loaded_events = int((cov_2025_df["status"] == "loaded").sum())
    missing_events = cov_2025_df[cov_2025_df["status"] != "loaded"]["event"].tolist()

    print(f"\n2025 Holdout Coverage: {loaded_events}/{total_scheduled} scheduled events loaded ({len(holdout_rows)} rows).")

    # 3. Fit frozen candidate model strictly on 2022-2024
    print("\nFitting frozen candidate model (calibrated_hgb with GroupKFold) on 2022-2024...")
    candidate_bundle = fit_candidate_first_stop_model(train_rows, architecture="calibrated_hgb")
    pipeline = candidate_bundle["pipeline"]

    # 4. Fit baselines strictly on 2022-2024
    print("Fitting comparison baselines on 2022-2024...")
    median_baseline = HistoricalMedianBaseline().fit(train_rows)
    empirical_hazard = EmpiricalHazardBaseline().fit(train_rows)
    marginal_baseline = MarginalHazardBaseline().fit(train_rows)

    if holdout_rows.empty:
        print("No holdout rows available for 2025 evaluation.")
        return

    # 5. Evaluate on 2025 holdout
    print("Evaluating on 2025 holdout data...")
    X_test = holdout_rows[FEATURE_COLUMNS]
    y_test = holdout_rows["StopNextLap"].astype(int).to_numpy()
    p_candidate = pipeline.predict_proba(X_test)[:, 1]
    p_clipped = np.clip(p_candidate, 1e-8, 1 - 1e-8)

    candidate_brier = float(brier_score_loss(y_test, p_candidate))
    candidate_logloss = float(log_loss(y_test, p_clipped, labels=[0, 1]))
    candidate_auc = float(roc_auc_score(y_test, p_candidate)) if len(np.unique(y_test)) == 2 else None

    empirical_eval = empirical_hazard.evaluate(holdout_rows)
    marginal_eval = marginal_baseline.evaluate(holdout_rows)
    median_eval = median_baseline.evaluate(holdout_rows)

    calibration_diag = compute_calibration_diagnostics(y_test, p_candidate)

    # Lap MAE & intervals
    candidate_errors, baseline_median_errors, interval_hits = [], [], []
    event_groups = {}

    for (event_name, _), group in holdout_rows.groupby(["EventName", "Driver"], dropna=False):
        group = group.sort_values("LapNumber")
        positives = group[group["StopNextLap"] == 1]
        if positives.empty:
            continue
        first = group.iloc[0]
        event_str = str(first["EventName"])
        compound = str(first.get("Compound", "MEDIUM"))
        actual_lap = int(positives["LapNumber"].iloc[0]) + 1

        # Baseline error computed here but appended with the candidate error below,
        # so baseline and candidate populations always contain the same drivers.
        base_lap = median_baseline.predict_lap(event_str, compound)
        base_err = abs(base_lap - actual_lap)

        driver_state = {
            "driver": str(first["Driver"]),
            "team": str(first.get("Team", "Unknown")),
            "grid_position": first.get("GridPosition"),
            "starting_compound": compound,
            "starting_tyre_age": max(1, int(first.get("TyreLife", 1))),
        }
        race_distance = max(5, int(first.get("RaceLaps", group["LapNumber"].max())))
        forecast_rows = [
            _lap_features(driver_state, lap, race_distance, event_str, "pre-race")
            for lap in range(2, race_distance + 1)
        ]
        if not forecast_rows:
            continue
        f_df = pd.DataFrame(forecast_rows)[FEATURE_COLUMNS]
        gp = pipeline.predict_proba(f_df)[:, 1]
        surv = 1.0
        masses = []
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
        cand_err = abs(pred_lap - actual_lap)
        candidate_errors.append(cand_err)
        baseline_median_errors.append(base_err)  # paired 1:1 with the candidate error

        p10 = int(laps[min(np.searchsorted(cdf, 0.10), len(laps) - 1)])
        p90 = int(laps[min(np.searchsorted(cdf, 0.90), len(laps) - 1)])
        hit = bool(p10 <= actual_lap <= p90)
        interval_hits.append(hit)

        if event_str not in event_groups:
            event_groups[event_str] = {"cand_errors": [], "base_errors": [], "hits": []}
        event_groups[event_str]["cand_errors"].append(cand_err)
        event_groups[event_str]["base_errors"].append(base_err)
        event_groups[event_str]["hits"].append(hit)

    # Candidate, baseline-median, and interval samples must describe the same drivers.
    assert len(candidate_errors) == len(baseline_median_errors) == len(interval_hits), (
        "Evaluation population mismatch: candidate, baseline-median, and interval "
        f"samples must align (got {len(candidate_errors)}, {len(baseline_median_errors)}, "
        f"{len(interval_hits)})."
    )

    candidate_lap_mae = float(np.mean(candidate_errors)) if candidate_errors else None
    baseline_lap_mae = float(np.mean(baseline_median_errors)) if baseline_median_errors else None
    coverage_10_90 = float(np.mean(interval_hits)) if interval_hits else None

    per_event_metrics = {}
    for ev_name, mdict in event_groups.items():
        ev_sub = holdout_rows[holdout_rows["EventName"] == ev_name]
        ev_y = ev_sub["StopNextLap"].astype(int).to_numpy()
        ev_p = pipeline.predict_proba(ev_sub[FEATURE_COLUMNS])[:, 1]
        per_event_metrics[ev_name] = {
            "drivers_evaluated": len(mdict["cand_errors"]),
            "candidate_lap_mae": float(np.mean(mdict["cand_errors"])) if mdict["cand_errors"] else None,
            "baseline_median_lap_mae": float(np.mean(mdict["base_errors"])) if mdict["base_errors"] else None,
            "interval_coverage_10_90": float(np.mean(mdict["hits"])) if mdict["hits"] else None,
            "brier_score": float(brier_score_loss(ev_y, ev_p)) if len(ev_y) else None,
        }

    status_str = "incomplete" if loaded_events < total_scheduled else "evaluated"

    backtest_report = {
        "status": status_str,
        "evaluation_type": "2025_final_holdout_frozen_model",
        "holdout_year": 2025,
        "total_scheduled_events": total_scheduled,
        "loaded_events": loaded_events,
        "missing_events": missing_events,
        "certification_note": (
            "Final certification remains incomplete because 2025 historical coverage is partial "
            f"({loaded_events}/{total_scheduled} events recovered within execution session). "
            "Model parameters and calibration were strictly frozen on 2022-2024 data without tuning on 2025."
        ),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_positive_rows": int(y_test.sum()),
        "drivers_with_observed_first_stop": len(candidate_errors),
        "metrics": {
            "candidate_model": {
                "brier_score": candidate_brier,
                "log_loss": candidate_logloss,
                "roc_auc": candidate_auc,
                "first_stop_lap_mae": candidate_lap_mae,
                "interval_coverage_10_90": coverage_10_90,
                "mean_predicted_hazard": float(np.mean(p_candidate)),
                "observed_stop_rate": float(y_test.mean()),
                "expected_calibration_error": calibration_diag["expected_calibration_error"],
            },
            "baseline_1_historical_median": {
                "first_stop_lap_mae": baseline_lap_mae,
                "drivers_evaluated": len(baseline_median_errors),
            },
            "baseline_2_empirical_hazard": {
                "brier_score": empirical_eval["brier_score"],
                "log_loss": empirical_eval["log_loss"],
            },
            "baseline_3_marginal_hazard": {
                "brier_score": marginal_eval["brier_score"],
                "log_loss": marginal_eval["log_loss"],
                "marginal_rate": marginal_eval["marginal_rate"],
            },
        },
        "calibration_details": calibration_diag["binned_calibration"],
        "quantile_calibration": calibration_diag.get("quantile_calibration", []),
        "calibration_notes": calibration_diag.get("calibration_notes", ""),
        "evaluation_qualifications": {
            "probabilistic_metrics": "Brier score, log loss and ROC-AUC are measured over all eligible per-lap hazard rows in the recovered holdout set.",
            "expected_calibration_error": "ECE is a count-weighted error over uniform probability bins; because per-lap hazards concentrate below 0.20 it is dominated by the lowest bin. See quantile_calibration for evenly populated reliability bins.",
            "stop_lap_mae": "Stop-lap MAE is conditional on an observed first stop; drivers without an observed first stop are right-censored and excluded. Baseline and candidate MAE are measured on the identical driver set.",
            "interval_coverage_10_90": "10th-90th interval coverage is conditional on observed stopping and uses the per-driver stop-lap distribution renormalised to sum to 1 within race distance.",
            "coverage_vs_nominal": "Empirical 10-90 coverage above the 80% nominal target indicates overcoverage (conservative/wide intervals), not exact calibration.",
            "season_completeness": f"Partial 2025 holdout: {loaded_events} of {total_scheduled} rounds ingested locally. Evaluation remains incomplete and is not a full-season result.",
        },
        "per_event_metrics": per_event_metrics,
    }

    report_path = REPORT_DIR / "candidate_first_stop_backtest_2025_holdout.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(backtest_report, f, indent=2, default=_json_default)
    print(f"\nSaved 2025 holdout backtest report: {report_path}")

    # Save frozen model artifact separately
    save_candidate_model(candidate_bundle, {"backtest_2025_holdout": backtest_report}, version="2025_holdout")

    print("\n" + "=" * 60)
    print("  2025 FROZEN HOLDOUT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Holdout Status: {status_str} ({loaded_events}/{total_scheduled} events)")
    print(f"  Candidate Lap MAE:       {candidate_lap_mae:.2f} laps" if candidate_lap_mae is not None else "  Candidate Lap MAE: N/A")
    print(f"  Historical Median MAE:   {baseline_lap_mae:.2f} laps" if baseline_lap_mae is not None else "  Historical Median MAE: N/A")
    print(f"  10th-90th Interval Cov:  {coverage_10_90:.1%}" if coverage_10_90 is not None else "")
    print(f"  Candidate Brier Score:   {candidate_brier:.4f} (Log loss: {candidate_logloss:.4f})")
    print(f"  Empirical Brier Score:   {empirical_eval['brier_score']:.4f}")
    print(f"  Marginal Brier Score:    {marginal_eval['brier_score']:.4f}")
    print(f"  Candidate ROC-AUC:       {candidate_auc:.4f}" if candidate_auc is not None else "")
    print(f"  Expected Calib Error:    {calibration_diag['expected_calibration_error']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_2025_holdout_evaluation()
