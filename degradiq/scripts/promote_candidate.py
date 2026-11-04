"""Promote a validated first-stop candidate to a tracked, served artifact.

Promotion is explicit, auditable and reversible. It copies a validated candidate
pipeline to a NEW tracked artifact (``first_stop_model_v2_hgb.joblib``) with a
label-versioned metadata sidecar, after verifying — via ``PredictionEngine`` — that
its feature schema and label definition are current. It **never** overwrites the
production logistic-regression model (``first_stop_model.joblib``); the app prefers
the promoted model when present and falls back to the protected LR if the promoted
file is removed.

Requires the validated candidate to be present on disk (it lives in the git-ignored
v2 run directory). Run locally, then commit the resulting tracked artifact + metadata.

    python -m degradiq.scripts.promote_candidate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.pit_strategy import PACKAGE_DIR, FEATURE_COLUMNS, LABEL_DEFINITION_VERSION
from degradiq.prediction_engine import (
    PredictionEngine, PROMOTED_MODEL_PATH, PROMOTED_METADATA_PATH, IncompatibleModelError,
)

RUN_DIR = PACKAGE_DIR / "cache" / "first_stop_v2_runs" / "run_v2_20261018"
DEFAULT_CANDIDATE = RUN_DIR / "candidate_2025_partial.joblib"
DEFAULT_CAND_META = RUN_DIR / "candidate_2025_partial_metadata.json"


def promote(candidate: Path, candidate_meta: Path) -> dict:
    if not candidate.exists():
        raise SystemExit(f"Validated candidate not found: {candidate} (it is git-ignored; "
                         "run the v2 pipeline/reproduction to produce it).")
    src_meta = json.loads(candidate_meta.read_text(encoding="utf-8")) if candidate_meta.exists() else {}

    # Gate 1: the candidate must load and be compatible with the CURRENT contract.
    engine = PredictionEngine(candidate, candidate_meta, require_current_label_version=True)
    prov = engine.provenance()
    if prov["feature_columns"] != FEATURE_COLUMNS:
        raise IncompatibleModelError("Candidate feature schema does not match current.")
    if prov["label_version_status"] != "current":
        raise IncompatibleModelError(
            f"Candidate label version is {prov['label_version_status']}, not current.")

    # Gate 2: the recorded hash must match the bytes being promoted (no drift).
    disk_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    recorded = src_meta.get("model_sha256")
    if recorded and recorded != disk_sha:
        raise SystemExit(f"Candidate sha256 drift: metadata {recorded[:12]} != disk {disk_sha[:12]}")

    PROMOTED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, PROMOTED_MODEL_PATH)  # byte-identical: preserves the hash

    metadata = {
        "model_file": PROMOTED_MODEL_PATH.name,
        "feature_columns": FEATURE_COLUMNS,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "architecture": src_meta.get("architecture", "calibrated_hgb"),
        "training_seasons": src_meta.get("training_seasons"),
        "training_rows": src_meta.get("training_rows"),
        "model_sha256": disk_sha,
        "trained_at_utc": src_meta.get("trained_at_utc"),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_from": {
            "run_id": RUN_DIR.name,
            "candidate": candidate.name,
            "tag": src_meta.get("tag"),
        },
        "evaluation": {
            "temporal_and_loco": "reports/experiments/first_stop_v2/generalization_summary.json",
            "backtest": "reports/experiments/first_stop_v2/backtest_2025_partial.json",
            "model_card": "reports/experiments/first_stop_v2/MODEL_CARD.md",
        },
        "note": "Promoted validated candidate. The production logistic-regression model "
                "(first_stop_model.joblib) is unchanged; remove this file to revert.",
    }
    PROMOTED_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Gate 3: the promoted artifact must load and serve as the current model.
    served = PredictionEngine(PROMOTED_MODEL_PATH, PROMOTED_METADATA_PATH,
                              require_current_label_version=True)
    assert served.provenance()["model_sha256"] == disk_sha

    print("Promoted candidate ->", PROMOTED_MODEL_PATH)
    print("  sha256:", disk_sha)
    print("  label :", LABEL_DEFINITION_VERSION, "| seasons:", metadata["training_seasons"])
    print("  production LR (first_stop_model.joblib) left untouched.")
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    p.add_argument("--candidate-metadata", type=Path, default=DEFAULT_CAND_META)
    args = p.parse_args()
    promote(args.candidate, args.candidate_metadata)


if __name__ == "__main__":
    main()
