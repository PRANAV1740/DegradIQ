"""Regression tests for candidate promotion.

Verify that, when a promoted validated artifact is present, the served engine uses it
(label version 'current', calibrated_hgb), that the production logistic-regression
model is untouched and still classifies as legacy, and that the promoted metadata
records auditable provenance.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from degradiq.pit_strategy import FEATURE_COLUMNS, LABEL_DEFINITION_VERSION, MODEL_PATH
from degradiq.prediction_engine import (
    load_default_engine, PredictionEngine, PROMOTED_MODEL_PATH, PROMOTED_METADATA_PATH,
)

PROMOTED = Path(PROMOTED_MODEL_PATH).exists()


@unittest.skipUnless(PROMOTED, "no promoted artifact present")
class PromotionTests(unittest.TestCase):
    def test_served_engine_is_the_promoted_model(self):
        prov = load_default_engine().provenance()
        self.assertTrue(prov["model_path"].endswith("first_stop_model_v2_hgb.joblib"))
        self.assertEqual(prov["label_version_status"], "current")
        self.assertEqual(prov["label_definition_version"], LABEL_DEFINITION_VERSION)
        self.assertEqual(prov["architecture"], "calibrated_hgb")
        self.assertEqual(prov["feature_columns"], FEATURE_COLUMNS)

    def test_production_lr_untouched_and_still_legacy(self):
        # The protected production model must remain loadable and unversioned.
        self.assertTrue(Path(MODEL_PATH).exists())
        lr = PredictionEngine(MODEL_PATH)
        self.assertEqual(lr.provenance()["label_version_status"], "unversioned_legacy")

    def test_require_current_label_version_now_satisfiable(self):
        # With the promoted model present, the strict gate succeeds via the default.
        eng = load_default_engine(require_current_label_version=True)
        self.assertIsNotNone(eng)
        self.assertEqual(eng.provenance()["label_version_status"], "current")

    def test_promoted_metadata_records_provenance(self):
        meta = json.loads(Path(PROMOTED_METADATA_PATH).read_text(encoding="utf-8"))
        self.assertEqual(meta["label_definition_version"], LABEL_DEFINITION_VERSION)
        self.assertEqual(len(meta["model_sha256"]), 64)
        self.assertIn("promoted_from", meta)
        self.assertIn("evaluation", meta)
        self.assertEqual(meta["feature_columns"], FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
