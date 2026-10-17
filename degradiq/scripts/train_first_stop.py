"""Download completed historical races, validate chronologically, and train candidate first-stop model.

Run from repository root:
    python -m degradiq.scripts.train_first_stop --start-year 2022 --end-year 2024 --holdout-year 2024

Follows DegradIQ Phase 3, 4, 5, 6 requirements:
- Outbound request pacing and backoff via RateLimiter.
- Atomic session checkpoints under degradiq/cache/hazard_checkpoints/.
- Zero leakage: candidate model and baselines fit strictly on pre-holdout data.
- Separate candidate artifacts; preserves production first_stop_model.joblib.
- Side-by-side evaluation against historical median, empirical hazard, and marginal baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any
import warnings

import fastf1
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.config import YEAR
from degradiq.rate_limiter import RateLimiter, install_rate_limiter, CheckpointManager
from degradiq.evaluation import (
    HistoricalMedianBaseline,
    EmpiricalHazardBaseline,
    MarginalHazardBaseline,
    compute_calibration_diagnostics,
)
from degradiq.pit_strategy import (
    PACKAGE_DIR, FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    build_hazard_rows, fit_first_stop_model, _lap_features,
)

logger = logging.getLogger("degradiq.train_first_stop")
CACHE = PACKAGE_DIR / "cache" / "fastf1"
REPORT_DIR = PACKAGE_DIR / "reports"
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"


def _event_date(row: pd.Series):
    value = row.get("EventDate", row.get("Session5Date", pd.NaT))
    return pd.to_datetime(value, errors="coerce", utc=True)


def collect_rows_with_checkpoints(
    start_year: int,
    end_year: int,
    checkpoint_mgr: CheckpointManager,
    limiter: RateLimiter,
    offline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect hazard rows across seasons using checkpoints and rate-paced ingestion."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    install_rate_limiter(limiter)

    all_rows, statuses = [], []
    now = pd.Timestamp.now(tz="UTC")

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error("Schedule unavailable for %d: %s", year, exc)
            statuses.append({
                "year": year, "round": "*", "event": "*", "status": "schedule unavailable",
                "rows": 0, "positive_rows": 0, "negative_rows": 0, "censored_rows": 0,
                "error": f"{type(exc).__name__}: {exc}"
            })
            continue

        for _, event in schedule.iterrows():
            name = str(event.get("EventName", "Unknown"))
            round_id = event.get("RoundNumber", name)
            if pd.isna(round_id) or round_id == 0:
                round_id = name

            event_date = _event_date(event)
            if pd.notna(event_date) and event_date > now:
                statuses.append({
                    "year": year, "round": str(round_id), "event": name, "status": "future/pending",
                    "rows": 0, "positive_rows": 0, "negative_rows": 0, "censored_rows": 0, "error": ""
                })
                continue

            # 1. Check existing durable checkpoint first
            if checkpoint_mgr.has_valid_checkpoint(year, round_id, FEATURE_COLUMNS, event_name=name):
                df_cp, meta = checkpoint_mgr.load_checkpoint(year, round_id, event_name=name)
                status = meta.get("status", "loaded")
                pos_count = meta.get("positive_count", 0)
                row_count = meta.get("row_count", len(df_cp))
                neg_count = row_count - pos_count
                if not df_cp.empty:
                    df_cp["Season"] = year
                    all_rows.append(df_cp)
                statuses.append({
                    "year": year, "round": str(round_id), "event": name, "status": status,
                    "rows": row_count, "positive_rows": pos_count, "negative_rows": neg_count,
                    "censored_rows": 0, "error": meta.get("error", "")
                })
                print(f"[Checkpoint] {year} Round {round_id} ({name}): {row_count} rows, {pos_count} stops")
                continue

            # 2. Checkpoint missing: if offline, record skip; otherwise load session with rate-limiter protection
            if offline:
                statuses.append({
                    "year": year, "round": str(round_id), "event": name, "status": "unavailable (offline mode)",
                    "rows": 0, "positive_rows": 0, "negative_rows": 0, "censored_rows": 0, "error": "Offline mode"
                })
                continue

            try:
                print(f"[Ingesting] {year} Round {round_id} ({name})...")
                session = fastf1.get_session(year, round_id, "R")
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
                results = getattr(session, "results", pd.DataFrame())
                max_lap = pd.to_numeric(laps.get("LapNumber", pd.Series(dtype=float)), errors="coerce").max()
                race_distance = int(max_lap) if pd.notna(max_lap) else None
                rows = build_hazard_rows(laps, results, name, race_distance)

                if not rows.empty:
                    rows["Season"] = year
                    all_rows.append(rows)
                    pos_count = int(rows["StopNextLap"].sum())
                    neg_count = len(rows) - pos_count
                    status = "loaded"
                    checkpoint_mgr.save_checkpoint(year, round_id, name, rows, status=status)
                else:
                    pos_count = 0
                    neg_count = 0
                    status = "no eligible rows"
                    checkpoint_mgr.save_checkpoint(year, round_id, name, pd.DataFrame(), status=status)

                statuses.append({
                    "year": year, "round": str(round_id), "event": name, "status": status,
                    "rows": len(rows), "positive_rows": pos_count, "negative_rows": neg_count,
                    "censored_rows": 0, "error": ""
                })
                print(f"[Loaded] {year} {name}: {len(rows)} rows, {pos_count} stop labels")
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                logger.warning("Failed session for %d %s: %s", year, name, err_msg)
                checkpoint_mgr.save_checkpoint(year, round_id, name, pd.DataFrame(), status="unavailable", error=err_msg)
                statuses.append({
                    "year": year, "round": str(round_id), "event": name, "status": "unavailable",
                    "rows": 0, "positive_rows": 0, "negative_rows": 0, "censored_rows": 0, "error": err_msg
                })

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return combined, pd.DataFrame(statuses)


def fit_candidate_first_stop_model(
    training_rows: pd.DataFrame,
    architecture: str = "calibrated_hgb",
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit calibrated candidate hazard model on training rows with event-grouped cross-validation."""
    if training_rows is None or training_rows.empty:
        raise ValueError("No first-stop training rows provided.")
    data = training_rows.dropna(subset=["StopNextLap"]).copy()
    data["StopNextLap"] = pd.to_numeric(data["StopNextLap"], errors="coerce")
    data = data.dropna(subset=["StopNextLap"])
    positives = int((data["StopNextLap"] == 1).sum())
    negatives = int((data["StopNextLap"] == 0).sum())
    if positives < 10 or negatives < 100:
        raise ValueError(f"Insufficient training data: {positives} positive stops, {negatives} non-stop rows.")

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

    if architecture == "calibrated_hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        pre_hgb = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
        ])
        base_model = Pipeline([
            ("preprocessor", pre_hgb),
            ("hazard", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12, min_samples_leaf=40, l2_regularization=3.0, random_state=42)),
        ])
    else:
        # Default: Calibrated Logistic Regression (Sigmoid, CV=5)
        base_model = Pipeline([
            ("preprocessor", preprocessor),
            ("hazard", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ])

    if groups is None and "EventName" in data.columns:
        groups = data["EventName"].astype(str).to_numpy()

    unique_groups = len(np.unique(groups)) if groups is not None else 0
    if unique_groups >= 5:
        from sklearn.model_selection import GroupKFold
        cv_splits = list(GroupKFold(n_splits=5).split(data[FEATURE_COLUMNS], data["StopNextLap"].astype(int), groups=groups))
        calibrated_pipeline = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_splits)
    else:
        calibrated_pipeline = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
    calibrated_pipeline.fit(data[FEATURE_COLUMNS], data["StopNextLap"].astype(int))

    # Derive the training seasons directly from the data actually fitted so saved
    # metadata can never drift from the rows the model was trained on.
    if "Season" in data.columns:
        training_seasons = sorted(
            int(s) for s in pd.to_numeric(data["Season"], errors="coerce").dropna().unique()
        )
    else:
        training_seasons = []

    return {
        "pipeline": calibrated_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": int(len(data)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_rate": float(positives / len(data)),
        "architecture": architecture,
        "training_seasons": training_seasons,
    }


def evaluate_candidate_chronological(
    rows: pd.DataFrame,
    coverage_df: pd.DataFrame,
    holdout_year: int = 2024,
    architecture: str = "calibrated_hgb",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly train on Season < holdout_year and evaluate on holdout_year against 3 baselines."""
    train = rows[rows["Season"] < holdout_year].copy()
    test = rows[rows["Season"] == holdout_year].copy()

    # Check holdout coverage completeness
    holdout_coverage = coverage_df[coverage_df["year"] == holdout_year]
    total_scheduled = int(len(holdout_coverage))
    loaded_events = int((holdout_coverage["status"] == "loaded").sum())
    missing_events = holdout_coverage[holdout_coverage["status"] != "loaded"]["event"].tolist()

    if train.empty or test.empty:
        status_val = "incomplete" if loaded_events < total_scheduled else "not available"
        report = {
            "status": status_val,
            "reason": f"Need both pre-{holdout_year} training rows and {holdout_year} holdout rows.",
            "train_rows": len(train),
            "holdout_rows": len(test),
            "total_scheduled_events": total_scheduled,
            "loaded_events": loaded_events,
            "missing_events": missing_events,
        }
        return {}, report

    # 1. Fit Candidate Model on Training Set ONLY
    bundle = fit_candidate_first_stop_model(train, architecture=architecture)
    pipeline = bundle["pipeline"]

    # 2. Fit Baselines on Training Set ONLY
    median_baseline = HistoricalMedianBaseline().fit(train)
    empirical_hazard = EmpiricalHazardBaseline().fit(train)
    marginal_baseline = MarginalHazardBaseline().fit(train)

    # 3. Test Set Evaluation
    X_test = test[FEATURE_COLUMNS]
    y_test = test["StopNextLap"].astype(int).to_numpy()
    p_candidate = pipeline.predict_proba(X_test)[:, 1]
    p_clipped = np.clip(p_candidate, 1e-8, 1 - 1e-8)

    # Probabilistic metrics
    candidate_brier = float(brier_score_loss(y_test, p_candidate))
    candidate_logloss = float(log_loss(y_test, p_clipped, labels=[0, 1]))
    candidate_auc = float(roc_auc_score(y_test, p_candidate)) if len(np.unique(y_test)) == 2 else None

    empirical_eval = empirical_hazard.evaluate(test)
    marginal_eval = marginal_baseline.evaluate(test)
    median_eval = median_baseline.evaluate(test)

    # Calibration diagnostics
    calibration_diag = compute_calibration_diagnostics(y_test, p_candidate)

    # 4. Simulation of Stop-Lap Horizon (Lap MAE & 10th-90th Interval Coverage)
    candidate_errors = []
    interval_hits = []
    baseline_median_errors = []
    event_groups: dict[str, dict[str, list[Any]]] = {}

    for (event_name, _), group in test.groupby(["EventName", "Driver"], dropna=False):
        group = group.sort_values("LapNumber")
        positives = group[group["StopNextLap"] == 1]
        if positives.empty:
            continue  # Right-censored driver: exclude from observed stop-lap MAE

        first = group.iloc[0]
        event_name_str = str(first["EventName"])
        starting_compound = str(first.get("Compound", "MEDIUM"))
        actual_lap = int(positives["LapNumber"].iloc[0]) + 1

        # Historical median baseline error. Computed here but only appended in the
        # same block as the candidate error below, so the baseline and candidate
        # populations always contain exactly the same drivers even if a later
        # eligibility guard skips this driver's candidate forecast.
        base_lap = median_baseline.predict_lap(event_name_str, starting_compound)
        base_err = abs(base_lap - actual_lap)

        # Candidate model pre-race forecast horizon simulation
        driver_state = {
            "driver": str(first["Driver"]),
            "team": str(first.get("Team", "Unknown")),
            "grid_position": first.get("GridPosition"),
            "starting_compound": starting_compound,
            "starting_tyre_age": max(1, int(first.get("TyreLife", 1))),
        }
        race_distance = max(5, int(first.get("RaceLaps", group["LapNumber"].max())))
        forecast_rows = [
            _lap_features(driver_state, lap, race_distance, event_name_str, "pre-race")
            for lap in range(2, race_distance + 1)
        ]
        if not forecast_rows:
            continue

        f_df = pd.DataFrame(forecast_rows)[FEATURE_COLUMNS]
        group_p = pipeline.predict_proba(f_df)[:, 1]
        survival = 1.0
        masses = []
        laps = np.arange(2, race_distance + 1)
        for prob in group_p:
            masses.append(survival * float(prob))
            survival *= 1.0 - float(prob)
        masses = np.asarray(masses, dtype=float)
        if masses.sum() <= 0:
            continue
        masses /= masses.sum()
        cdf = np.cumsum(masses)

        # Median stop lap
        predicted_lap = int(laps[min(np.searchsorted(cdf, 0.50), len(laps) - 1)])
        cand_err = abs(predicted_lap - actual_lap)
        candidate_errors.append(cand_err)
        baseline_median_errors.append(base_err)  # paired 1:1 with the candidate error

        # 10th - 90th percentile interval
        p10_lap = int(laps[min(np.searchsorted(cdf, 0.10), len(laps) - 1)])
        p90_lap = int(laps[min(np.searchsorted(cdf, 0.90), len(laps) - 1)])
        hit = bool(p10_lap <= actual_lap <= p90_lap)
        interval_hits.append(hit)

        if event_name_str not in event_groups:
            event_groups[event_name_str] = {"cand_errors": [], "base_errors": [], "hits": []}
        event_groups[event_name_str]["cand_errors"].append(cand_err)
        event_groups[event_name_str]["base_errors"].append(base_err)
        event_groups[event_name_str]["hits"].append(hit)

    # Candidate, baseline-median, and interval samples must describe the same
    # drivers; a mismatch would make the reported MAE comparison invalid.
    assert len(candidate_errors) == len(baseline_median_errors) == len(interval_hits), (
        "Evaluation population mismatch: candidate, baseline-median, and interval "
        f"samples must align (got {len(candidate_errors)}, {len(baseline_median_errors)}, "
        f"{len(interval_hits)})."
    )

    candidate_lap_mae = float(np.mean(candidate_errors)) if candidate_errors else None
    baseline_lap_mae = float(np.mean(baseline_median_errors)) if baseline_median_errors else None
    coverage_10_90 = float(np.mean(interval_hits)) if interval_hits else None

    # Per-event evaluation
    per_event_metrics = {}
    for ev_name, metrics_dict in event_groups.items():
        ev_sub = test[test["EventName"] == ev_name]
        ev_y = ev_sub["StopNextLap"].astype(int).to_numpy()
        ev_p = pipeline.predict_proba(ev_sub[FEATURE_COLUMNS])[:, 1]
        per_event_metrics[ev_name] = {
            "drivers_evaluated": len(metrics_dict["cand_errors"]),
            "candidate_lap_mae": float(np.mean(metrics_dict["cand_errors"])) if metrics_dict["cand_errors"] else None,
            "baseline_median_lap_mae": float(np.mean(metrics_dict["base_errors"])) if metrics_dict["base_errors"] else None,
            "interval_coverage_10_90": float(np.mean(metrics_dict["hits"])) if metrics_dict["hits"] else None,
            "brier_score": float(brier_score_loss(ev_y, ev_p)) if len(ev_y) else None,
        }

    # Determine validation status
    if loaded_events < total_scheduled:
        overall_status = "incomplete"
    elif candidate_lap_mae is not None and baseline_lap_mae is not None and candidate_lap_mae > baseline_lap_mae:
        overall_status = "evaluated_not_better"
    else:
        overall_status = "evaluated"

    backtest_report = {
        "status": overall_status,
        "holdout_year": holdout_year,
        "total_scheduled_events": total_scheduled,
        "loaded_events": loaded_events,
        "missing_events": missing_events,
        "train_rows": len(train),
        "holdout_rows": len(test),
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
            "probabilistic_metrics": "Brier score, log loss and ROC-AUC are measured over all eligible per-lap hazard rows in the holdout set.",
            "expected_calibration_error": "ECE is a count-weighted error over uniform probability bins; because per-lap hazards concentrate below 0.20 it is dominated by the lowest bin. See quantile_calibration for evenly populated reliability bins.",
            "stop_lap_mae": "Stop-lap MAE is conditional on an observed first stop; drivers without an observed first stop are right-censored and excluded. Baseline and candidate MAE are measured on the identical driver set.",
            "interval_coverage_10_90": "10th-90th interval coverage is conditional on observed stopping and uses the per-driver stop-lap distribution renormalised to sum to 1 within race distance.",
            "coverage_vs_nominal": "Empirical 10-90 coverage above the 80% nominal target indicates overcoverage (conservative/wide intervals), not exact calibration.",
        },
        "per_event_metrics": per_event_metrics,
    }
    return bundle, backtest_report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_candidate_model(bundle: dict[str, Any], metadata_extra: dict[str, Any], version: str = "",
                         training_seasons: list[int] | None = None) -> None:
    """Save candidate model separately without touching production artifacts.

    `training_seasons`, when provided, overrides the value recorded in metadata.
    Otherwise the seasons derived by `fit_candidate_first_stop_model` (from the
    rows actually fitted) are used, falling back to [2022, 2023] only if neither
    is available. This prevents stale hardcoded season lists in saved metadata.
    """
    import joblib
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{version}" if version else ""
    candidate_model_path = ARTIFACT_DIR / f"candidate_first_stop_model{suffix}.joblib"
    candidate_meta_path = ARTIFACT_DIR / f"candidate_first_stop_model{suffix}_metadata.json"

    joblib.dump(bundle["pipeline"], candidate_model_path)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": bundle.get("training_rows"),
        "positive_rows": bundle.get("positive_rows"),
        "negative_rows": bundle.get("negative_rows"),
        "positive_rate": bundle.get("positive_rate"),
        "training_seasons": (
            training_seasons
            if training_seasons is not None
            else (bundle.get("training_seasons") or [2022, 2023])
        ),
        "model_file": candidate_model_path.name,
        "architecture": bundle.get("architecture", "calibrated_lr"),
    }
    metadata.update(metadata_extra)
    with open(candidate_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    logger.info("Saved candidate model: %s", candidate_model_path)
    logger.info("Saved candidate metadata: %s", candidate_meta_path)


def main():
    parser = argparse.ArgumentParser(description="DegradIQ Candidate First-Stop Model Training")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--holdout-year", type=int, default=2024)
    parser.add_argument("--version", type=str, default="",
                        help="Candidate model version suffix (default: empty).")
    parser.add_argument("--architecture", type=str, default="calibrated_hgb",
                        choices=["calibrated_lr", "calibrated_hgb"],
                        help="Model architecture for candidate (default: calibrated_hgb).")
    parser.add_argument("--min-interval", type=float, default=9.0,
                        help="Conservative pacing interval for outbound requests.")
    parser.add_argument("--offline", action="store_true",
                        help="Only process sessions that have local checkpoints without outbound requests.")
    args = parser.parse_args()

    limiter = RateLimiter(min_interval_s=args.min_interval)
    checkpoint_mgr = CheckpointManager()

    print(f"Collecting rows for seasons {args.start_year} to {args.end_year} (holdout={args.holdout_year}, offline={args.offline})...")
    rows, coverage = collect_rows_with_checkpoints(args.start_year, args.end_year, checkpoint_mgr, limiter, offline=args.offline)

    suffix = f"_{args.version}" if args.version else ""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_cov_path = REPORT_DIR / f"candidate_first_stop_training_coverage{suffix}.csv"
    coverage.to_csv(candidate_cov_path, index=False)
    print(f"Saved coverage report: {candidate_cov_path}")

    if rows.empty:
        raise SystemExit("No training rows collected.")

    # Chronological backtest and baseline comparison
    bundle, backtest = evaluate_candidate_chronological(
        rows, coverage, holdout_year=args.holdout_year, architecture=args.architecture
    )

    candidate_backtest_path = REPORT_DIR / f"candidate_first_stop_backtest{suffix}.json"
    with open(candidate_backtest_path, "w", encoding="utf-8") as f:
        json.dump(backtest, f, indent=2, default=_json_default)
    print(f"Saved candidate backtest report: {candidate_backtest_path}")

    print("\n" + "=" * 60)
    print("  CHRONOLOGICAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Holdout Status: {backtest.get('status')}")
    print(f"  Holdout Season: {backtest.get('holdout_year')}")
    print(f"  Scheduled Events: {backtest.get('total_scheduled_events')}, Loaded: {backtest.get('loaded_events')}")
    metrics = backtest.get("metrics", {})
    if metrics:
        cand_m = metrics.get("candidate_model", {})
        base1 = metrics.get("baseline_1_historical_median", {})
        base2 = metrics.get("baseline_2_empirical_hazard", {})
        base3 = metrics.get("baseline_3_marginal_hazard", {})
        print(f"  Candidate Lap MAE:       {cand_m.get('first_stop_lap_mae'):.2f} laps" if cand_m.get('first_stop_lap_mae') is not None else "  Candidate Lap MAE: N/A")
        print(f"  Historical Median MAE:   {base1.get('first_stop_lap_mae'):.2f} laps" if base1.get('first_stop_lap_mae') is not None else "  Historical Median MAE: N/A")
        print(f"  10th-90th Interval Cov:  {cand_m.get('interval_coverage_10_90'):.1%}" if cand_m.get('interval_coverage_10_90') is not None else "")
        print(f"  Candidate Brier Score:   {cand_m.get('brier_score'):.4f} (Log loss: {cand_m.get('log_loss'):.4f})")
        print(f"  Empirical Brier Score:   {base2.get('brier_score'):.4f} (Log loss: {base2.get('log_loss'):.4f})")
        print(f"  Marginal Brier Score:    {base3.get('brier_score'):.4f} (Log loss: {base3.get('log_loss'):.4f})")
        print(f"  Candidate ROC-AUC:       {cand_m.get('roc_auc'):.4f}" if cand_m.get('roc_auc') is not None else "")
        print(f"  Expected Calib Error:    {cand_m.get('expected_calibration_error'):.4f}")
    print("=" * 60 + "\n")

    if bundle:
        save_candidate_model(bundle, {"chronological_backtest": backtest}, version=args.version)


if __name__ == "__main__":
    main()
