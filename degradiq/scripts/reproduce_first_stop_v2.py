"""Deterministic, offline reproduction of the first-stop v2 validation experiment.

This is the reconstructed orchestration entry point for the ``run_v2_20261018``
experiment whose original driver script was not committed. It does NOT re-derive
labels from raw FastF1 sessions and performs NO network access: it consumes the
*certified* per-event dataset and coverage already produced by that run (the
cached, label-versioned hazard rows) and re-runs the committed training and
evaluation logic (:func:`evaluate_candidate_chronological`) exactly as the
original run did — once per chronological holdout (2024 development, 2025 partial).

Why this is a faithful reconstruction, not a re-modelling:
    * It reuses the frozen label definition and feature schema
      (``LABEL_DEFINITION_VERSION`` / ``FEATURE_COLUMNS``) via the committed code.
    * It reuses the exact certified rows (``combined_certified.parquet``) and
      coverage (``coverage.csv``) from the source run, so the training/evaluation
      splits (Season < holdout vs Season == holdout) are identical.
    * It reuses the committed model configuration and calibration procedure
      (calibrated HGB + GroupKFold(EventName) calibration) through
      ``fit_candidate_first_stop_model`` inside ``evaluate_candidate_chronological``.
    * Nothing in the modelling path is altered to change the metrics.

Outputs are written to a NEW, isolated verification run directory; the original
run and all production/legacy artifacts are never touched. A machine-checkable
``reproduction_report.json`` compares regenerated metrics against the historical
v2 outputs within declared numerical tolerances, and records each regenerated
model's SHA-256 beside the historical hash.

Usage (offline, cached data only)::

    python -m degradiq.scripts.reproduce_first_stop_v2

    # or against an explicit source/destination:
    python -m degradiq.scripts.reproduce_first_stop_v2 \
        --source-run degradiq/cache/first_stop_v2_runs/run_v2_20261018 \
        --out-run    degradiq/cache/first_stop_v2_runs/run_v2_verify_<id>
"""
from __future__ import annotations

import argparse
import hashlib
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

import joblib  # noqa: E402
import sklearn  # noqa: E402

from degradiq.pit_strategy import (  # noqa: E402
    PACKAGE_DIR,
    FEATURE_COLUMNS,
    LABEL_DEFINITION_VERSION,
)
from degradiq.rate_limiter import DIAGNOSTICS_SCHEMA_VERSION  # noqa: E402
from degradiq.scripts.train_first_stop import (  # noqa: E402
    evaluate_candidate_chronological,
    _json_default,
)

V2_RUNS_DIR = PACKAGE_DIR / "cache" / "first_stop_v2_runs"
DEFAULT_SOURCE_RUN = V2_RUNS_DIR / "run_v2_20261018"
TRACKED_PROVENANCE_DIR = PACKAGE_DIR / "reports" / "experiments" / "first_stop_v2"

# The canonical v2 candidates and the chronological holdout each was evaluated on.
# Each entry drives one reproduction of the committed chronological backtest.
CANDIDATES = (
    {"tag": "2024_dev", "holdout_year": 2024,
     "historical_backtest": "backtest_2024_dev.json",
     "historical_metadata": "candidate_2024_dev_metadata.json"},
    {"tag": "2025_partial", "holdout_year": 2025,
     "historical_backtest": "backtest_2025_partial.json",
     "historical_metadata": "candidate_2025_partial_metadata.json"},
)

# Metrics compared against the historical run, with the declared absolute tolerance.
# Library versions are pinned to the recorded manifest, so bit-exact reproduction is
# expected; the tolerance guards only against platform-level floating-point noise.
_METRIC_ATOL = 1e-4
_METRIC_KEYS = (
    "brier_score", "log_loss", "roc_auc", "first_stop_lap_mae",
    "interval_coverage_10_90", "expected_calibration_error",
)
_COUNT_KEYS = ("train_rows", "holdout_rows", "holdout_positive_rows", "loaded_events")


def _git_revision() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconciliation_counts(reconciliation_csv: Path) -> dict[str, dict[str, int]]:
    """Re-derive the three reconciliation summaries from the preserved audit CSV.

    The per-event comparison (``reconciliation_audit.csv``) is a preserved input of
    the source run. The three historically reported "identical/different" counts are
    three *definitions* applied to that same table; this reproduces each count so the
    apparent contradiction (55/24 vs 31/48 vs 28/51) is shown to be one dataset under
    three comparison lenses, not inconsistent evidence.
    """
    df = pd.read_csv(reconciliation_csv)
    n = len(df)

    def split(mask: pd.Series) -> dict[str, int]:
        ident = int(mask.sum())
        return {"identical": ident, "different": int(n - ident), "events": n}

    feature_plus_target = split((df["feat_mm"] == 0) & (df["tgt_mm"] == 0))
    row_membership = split((df["only_new"] == 0) & (df["only_legacy"] == 0))
    strict_auth = split(df["auth_status"].astype(str) == "identical")
    return {
        "feature_plus_target": feature_plus_target,   # historically 55 / 24
        "row_membership": row_membership,             # historically 31 / 48 (manifest)
        "strict_rows_features_target": strict_auth,   # historically 28 / 51 (auth_status)
    }


def _load_historical_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cm = data.get("metrics", {}).get("candidate_model", {})
    out = {k: cm.get(k) for k in _METRIC_KEYS}
    for k in _COUNT_KEYS:
        out[k] = data.get(k)
    return out


def _regenerated_metrics(report: dict[str, Any]) -> dict[str, Any]:
    cm = report.get("metrics", {}).get("candidate_model", {})
    out = {k: cm.get(k) for k in _METRIC_KEYS}
    for k in _COUNT_KEYS:
        out[k] = report.get(k)
    return out


def _compare(historical: dict[str, Any], regenerated: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    within = True
    for k in _METRIC_KEYS:
        h, r = historical.get(k), regenerated.get(k)
        if h is None or r is None:
            deltas[k] = {"historical": h, "regenerated": r, "abs_diff": None,
                         "within_tol": h == r}
            within = within and (h == r)
            continue
        diff = abs(float(h) - float(r))
        ok = diff <= _METRIC_ATOL
        within = within and ok
        deltas[k] = {"historical": h, "regenerated": r, "abs_diff": diff, "within_tol": ok}
    for k in _COUNT_KEYS:
        h, r = historical.get(k), regenerated.get(k)
        ok = h == r
        within = within and ok
        deltas[k] = {"historical": h, "regenerated": r, "exact_match": ok}
    return {"within_tolerance": within, "atol": _METRIC_ATOL, "fields": deltas}


def reproduce(source_run: Path, out_run: Path, promote_provenance: bool = True) -> dict[str, Any]:
    if not source_run.is_dir():
        raise SystemExit(f"Source run directory not found: {source_run}")
    combined_path = source_run / "combined_certified.parquet"
    coverage_path = source_run / "coverage.csv"
    for required in (combined_path, coverage_path):
        if not required.exists():
            raise SystemExit(
                f"Required cached input missing: {required}. This script does NOT "
                "download data; the certified dataset and coverage must already exist."
            )

    out_run.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(combined_path)
    coverage = pd.read_csv(coverage_path)

    # Fail-closed sanity: the certified dataset must carry the current label version.
    versions = set(pd.Series(rows.get("LabelDefVersion")).dropna().unique())
    if versions != {LABEL_DEFINITION_VERSION}:
        raise SystemExit(
            f"Certified dataset label version {sorted(versions)} != current "
            f"'{LABEL_DEFINITION_VERSION}'. Refusing to reproduce against a mismatched label."
        )

    run_id = out_run.name
    git_rev = _git_revision()
    libs = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }

    print("=" * 70)
    print(f"  REPRODUCTION  run_id={run_id}")
    print(f"  source      = {source_run}")
    print(f"  git         = {git_rev}")
    print(f"  label       = {LABEL_DEFINITION_VERSION}")
    print(f"  libs        = {libs}")
    print(f"  dataset     = {len(rows)} rows, {int(rows['StopNextLap'].sum())} positives")
    print("=" * 70)

    candidate_records: list[dict[str, Any]] = []
    all_within = True

    for spec in CANDIDATES:
        tag, holdout = spec["tag"], spec["holdout_year"]
        print(f"\n[{tag}] chronological holdout={holdout} ...")
        bundle, report = evaluate_candidate_chronological(
            rows, coverage, holdout_year=holdout, architecture="calibrated_hgb"
        )
        if not bundle:
            raise SystemExit(f"[{tag}] evaluation produced no model: {report}")

        # Persist regenerated model + backtest into the isolated verification run only.
        model_path = out_run / f"candidate_{tag}.joblib"
        joblib.dump(bundle["pipeline"], model_path)
        model_sha = _sha256_file(model_path)

        backtest_path = out_run / f"backtest_{tag}.json"
        backtest_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

        metadata = {
            "tag": tag,
            "holdout_year": holdout,
            "training_seasons": bundle.get("training_seasons"),
            "training_rows": bundle.get("training_rows"),
            "positive_rows": bundle.get("positive_rows"),
            "negative_rows": bundle.get("negative_rows"),
            "label_definition_version": LABEL_DEFINITION_VERSION,
            "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "architecture": bundle.get("architecture"),
            "feature_columns": FEATURE_COLUMNS,
            "libs": libs,
            "model_sha256": model_sha,
        }
        (out_run / f"candidate_{tag}_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

        historical = _load_historical_metrics(source_run / spec["historical_backtest"])
        regenerated = _regenerated_metrics(report)
        comparison = _compare(historical, regenerated)

        historical_sha = None
        hist_meta_path = source_run / spec["historical_metadata"]
        if hist_meta_path.exists():
            historical_sha = json.loads(hist_meta_path.read_text(encoding="utf-8")).get("model_sha256")
        sha_match = (historical_sha == model_sha)

        all_within = all_within and comparison["within_tolerance"]
        candidate_records.append({
            "tag": tag,
            "holdout_year": holdout,
            "model_sha256": model_sha,
            "historical_model_sha256": historical_sha,
            "model_sha256_match": sha_match,
            "metric_comparison": comparison,
        })

        print(f"  within tolerance (atol={_METRIC_ATOL}): {comparison['within_tolerance']}")
        print(f"  model sha256 byte-identical to historical: {sha_match}")
        for k in _METRIC_KEYS:
            f = comparison["fields"][k]
            print(f"    {k:28s} hist={f['historical']}  regen={f['regenerated']}  "
                  f"|diff|={f['abs_diff']}")

    recon = reconciliation_counts(source_run / "reconciliation_audit.csv")

    reproduction_report = {
        "run_id": run_id,
        "reproduced_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev,
        "source_run": str(source_run.relative_to(ROOT)) if source_run.is_relative_to(ROOT) else str(source_run),
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "libraries": libs,
        "seeds": {"candidate_random_state": 42, "note": "calibrated_hgb random_state is fixed in fit_candidate_first_stop_model"},
        "dataset": {
            "combined_rows": int(len(rows)),
            "combined_positives": int(rows["StopNextLap"].sum()),
            "per_season": {int(s): int(v) for s, v in rows.groupby("Season")["StopNextLap"].size().items()},
        },
        "reconciliation_counts": recon,
        "candidates": candidate_records,
        "all_metrics_within_tolerance": all_within,
    }
    (out_run / "reproduction_report.json").write_text(
        json.dumps(reproduction_report, indent=2, default=_json_default), encoding="utf-8")

    if promote_provenance:
        _write_tracked_provenance(source_run, reproduction_report)

    print("\n" + "=" * 70)
    print(f"  ALL METRICS WITHIN TOLERANCE: {all_within}")
    print(f"  verification run written to: {out_run}")
    print(f"  tracked provenance: {TRACKED_PROVENANCE_DIR}")
    print("=" * 70)
    return reproduction_report


def _write_tracked_provenance(source_run: Path, reproduction_report: dict[str, Any]) -> None:
    """Copy the canonical run's small, text-only provenance files into a tracked
    location, and write a consolidated provenance.json + CANONICAL.md.

    Binary artifacts (parquet, joblib) and the FastF1 cache are intentionally NOT
    copied here; they remain in the git-ignored run directory.
    """
    TRACKED_PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    copy_names = [
        "manifest.json",
        "backtest_2024_dev.json",
        "backtest_2025_partial.json",
        "candidate_2024_dev_metadata.json",
        "candidate_2025_partial_metadata.json",
        "reconciliation_audit.csv",
        "coverage.csv",
    ]
    copied = []
    for name in copy_names:
        src = source_run / name
        if src.exists():
            (TRACKED_PROVENANCE_DIR / name).write_bytes(src.read_bytes())
            copied.append(name)

    provenance = {
        "canonical_run_id": source_run.name,
        "canonical_candidates": [
            {"tag": c["tag"], "holdout_year": c["holdout_year"],
             "model_sha256": c["historical_model_sha256"]}
            for c in reproduction_report["candidates"]
        ],
        "label_definition_version": reproduction_report["label_definition_version"],
        "diagnostics_schema_version": reproduction_report["diagnostics_schema_version"],
        "feature_columns": reproduction_report["feature_columns"],
        "libraries": reproduction_report["libraries"],
        "reconciliation_counts": reproduction_report["reconciliation_counts"],
        "reproduction_entry_point": "python -m degradiq.scripts.reproduce_first_stop_v2",
        "last_verified_git_revision": reproduction_report["git_revision"],
        "last_verified_within_tolerance": reproduction_report["all_metrics_within_tolerance"],
        "preserved_files": copied,
        "superseded_committed_candidate": {
            "note": "degradiq/artifacts/candidate_first_stop_model* (trained 2026-10-15) and "
                    "degradiq/reports/candidate_first_stop_backtest*.json describe an EARLIER "
                    "candidate and are NOT the canonical v2 result. Use the candidates above.",
        },
    }
    (TRACKED_PROVENANCE_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=_json_default), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN,
                        help="Directory of the canonical v2 run to reproduce (read-only).")
    parser.add_argument("--out-run", type=Path, default=None,
                        help="Isolated verification output directory (default: a fresh timestamped dir).")
    parser.add_argument("--no-promote", action="store_true",
                        help="Do not refresh the tracked provenance files.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any metric falls outside the declared tolerance.")
    args = parser.parse_args()

    out_run = args.out_run or (V2_RUNS_DIR / f"run_v2_verify_{datetime.now():%Y%m%d_%H%M%S}")
    report = reproduce(args.source_run, out_run, promote_provenance=not args.no_promote)
    if args.strict and not report["all_metrics_within_tolerance"]:
        raise SystemExit("Reproduction FAILED: metrics outside declared tolerance.")


if __name__ == "__main__":
    main()
