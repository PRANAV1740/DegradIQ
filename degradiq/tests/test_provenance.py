"""P0 regression tests: experiment provenance and reproduction-run configuration.

These tests are fast and operate only on tracked, text-only provenance files and
on the pure helpers in ``reproduce_first_stop_v2``. They do NOT train models,
touch the git-ignored run directories, or require network access. Their job is to
keep the canonical candidate unambiguous and the recorded run configuration
consistent with the current code (label version, feature schema, diagnostics
schema), and to lock in the three reconciliation-count definitions.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from degradiq.pit_strategy import FEATURE_COLUMNS, LABEL_DEFINITION_VERSION
from degradiq.rate_limiter import DIAGNOSTICS_SCHEMA_VERSION
from degradiq.scripts import reproduce_first_stop_v2 as repro

PROV_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiments" / "first_stop_v2"


def _load(name: str) -> dict:
    return json.loads((PROV_DIR / name).read_text(encoding="utf-8"))


class ProvenanceFilesTests(unittest.TestCase):
    def test_tracked_provenance_files_present(self):
        for name in ("provenance.json", "manifest.json", "reconciliation_audit.csv",
                     "backtest_2024_dev.json", "backtest_2025_partial.json",
                     "candidate_2024_dev_metadata.json", "candidate_2025_partial_metadata.json",
                     "CANONICAL.md"):
            self.assertTrue((PROV_DIR / name).exists(), f"missing tracked provenance file: {name}")

    def test_provenance_matches_current_code_contract(self):
        prov = _load("provenance.json")
        self.assertEqual(prov["label_definition_version"], LABEL_DEFINITION_VERSION)
        self.assertEqual(prov["diagnostics_schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
        self.assertEqual(prov["feature_columns"], FEATURE_COLUMNS)

    def test_canonical_candidate_is_unambiguous(self):
        prov = _load("provenance.json")
        self.assertEqual(prov["canonical_run_id"], "run_v2_20261018")
        tags = {c["tag"] for c in prov["canonical_candidates"]}
        self.assertEqual(tags, {"2024_dev", "2025_partial"})
        # Every canonical candidate declares a non-empty SHA-256.
        for c in prov["canonical_candidates"]:
            self.assertTrue(c["model_sha256"] and len(c["model_sha256"]) == 64)
        # The earlier committed candidate is explicitly marked as superseded.
        self.assertIn("superseded_committed_candidate", prov)

    def test_canonical_sha_matches_tracked_metadata(self):
        prov = {c["tag"]: c["model_sha256"] for c in _load("provenance.json")["canonical_candidates"]}
        for tag in ("2024_dev", "2025_partial"):
            meta = _load(f"candidate_{tag}_metadata.json")
            self.assertEqual(meta["model_sha256"], prov[tag],
                             f"{tag}: provenance SHA-256 disagrees with tracked metadata")
            self.assertEqual(meta["label_definition_version"], LABEL_DEFINITION_VERSION)
            self.assertEqual(meta["feature_columns"], FEATURE_COLUMNS)

    def test_manifest_label_and_schema_consistent(self):
        man = _load("manifest.json")
        self.assertEqual(man["label_definition_version"], LABEL_DEFINITION_VERSION)
        self.assertEqual(man["diagnostics_schema_version"], DIAGNOSTICS_SCHEMA_VERSION)


class ReconciliationCountTests(unittest.TestCase):
    def test_three_definitions_reproduce_historical_counts(self):
        counts = repro.reconciliation_counts(PROV_DIR / "reconciliation_audit.csv")
        self.assertEqual((counts["feature_plus_target"]["identical"],
                          counts["feature_plus_target"]["different"]), (55, 24))
        self.assertEqual((counts["row_membership"]["identical"],
                          counts["row_membership"]["different"]), (31, 48))
        self.assertEqual((counts["strict_rows_features_target"]["identical"],
                          counts["strict_rows_features_target"]["different"]), (28, 51))
        # All three lenses describe the same 79-event table.
        self.assertEqual({c["events"] for c in counts.values()}, {79})

    def test_row_membership_matches_manifest(self):
        counts = repro.reconciliation_counts(PROV_DIR / "reconciliation_audit.csv")
        man = _load("manifest.json")
        self.assertEqual(counts["row_membership"]["identical"], man["reconciliation"]["identical"])
        self.assertEqual(counts["row_membership"]["different"], man["reconciliation"]["diff"])


class ToleranceComparisonTests(unittest.TestCase):
    def _hist(self):
        return {"brier_score": 0.0449, "log_loss": 0.1782, "roc_auc": 0.7436,
                "first_stop_lap_mae": 6.9631, "interval_coverage_10_90": 0.8403,
                "expected_calibration_error": 0.0184,
                "train_rows": 12604, "holdout_rows": 8549,
                "holdout_positive_rows": 407, "loaded_events": 24}

    def test_identical_metrics_within_tolerance(self):
        cmp = repro._compare(self._hist(), dict(self._hist()))
        self.assertTrue(cmp["within_tolerance"])

    def test_metric_drift_fails_tolerance(self):
        regen = dict(self._hist())
        regen["first_stop_lap_mae"] += 0.01  # 100x the declared atol
        cmp = repro._compare(self._hist(), regen)
        self.assertFalse(cmp["within_tolerance"])

    def test_count_mismatch_fails(self):
        regen = dict(self._hist())
        regen["holdout_rows"] += 1
        cmp = repro._compare(self._hist(), regen)
        self.assertFalse(cmp["within_tolerance"])


class EntryPointTests(unittest.TestCase):
    def test_entry_point_reuses_committed_evaluation_logic(self):
        # The reconstruction must delegate to committed training/evaluation, not a
        # re-implemented model path.
        src = (Path(__file__).resolve().parents[1] / "scripts" / "reproduce_first_stop_v2.py").read_text(encoding="utf-8")
        self.assertIn("from degradiq.scripts.train_first_stop import", src)
        self.assertIn("evaluate_candidate_chronological", src)
        # It must be offline: no FastF1 session loads or schedule fetches.
        self.assertNotIn("get_session", src)
        self.assertNotIn("get_event_schedule", src)


if __name__ == "__main__":
    unittest.main()
