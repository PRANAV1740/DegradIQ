"""Tests for the PredictionEngine interface (Phase IV).

Uses the tracked production model artifact (logistic-regression, unversioned legacy)
for the happy path, plus small synthetic artifacts for the incompatibility guards.
No network, no training of the real pipeline.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from degradiq.pit_strategy import FEATURE_COLUMNS, MODEL_PATH
from degradiq.prediction_engine import (
    PredictionEngine, IncompatibleModelError, load_default_engine,
)

PROD_AVAILABLE = Path(MODEL_PATH).exists()
DRIVERS = [
    {"driver": "VER", "grid_position": 1, "starting_compound": "MEDIUM", "starting_tyre_age": 1},
    {"driver": "NOR", "grid_position": 3, "starting_compound": "HARD", "starting_tyre_age": 1},
]


@unittest.skipUnless(PROD_AVAILABLE, "production model artifact not present")
class ProductionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PredictionEngine(MODEL_PATH)

    def test_loads_and_is_ready(self):
        self.assertTrue(self.engine.ready)
        self.assertEqual(self.engine.feature_columns, FEATURE_COLUMNS)

    def test_provenance_block(self):
        p = self.engine.provenance()
        self.assertEqual(len(p["model_sha256"]), 64)
        self.assertEqual(p["feature_columns"], FEATURE_COLUMNS)
        # Production model predates label versioning -> legacy, not current.
        self.assertEqual(p["label_version_status"], "unversioned_legacy")

    def test_predict_adds_interval_width_and_engine_block(self):
        out = self.engine.predict("United States Grand Prix", 56, DRIVERS,
                                  mode="pre-race", context_mode="historical")
        self.assertEqual(out["engine"]["context_mode"], "historical")
        self.assertEqual(out["engine"]["feature_mode"], "pre-race")
        self.assertIn("model", out["engine"])
        self.assertTrue(out["predictions"])
        for pred in out["predictions"]:
            self.assertIn("interval_width_if_stops", pred)
            w = pred["interval_width_if_stops"]
            self.assertTrue(w is None or w >= 0)
            self.assertIn("probability_stop_before_finish", pred)

    def test_require_current_label_version_rejects_legacy(self):
        with self.assertRaises(IncompatibleModelError):
            PredictionEngine(MODEL_PATH, require_current_label_version=True)

    def test_bad_feature_mode_raises(self):
        with self.assertRaises(ValueError):
            self.engine.predict("X", 50, DRIVERS, mode="bogus")

    def test_bad_context_mode_raises(self):
        with self.assertRaises(ValueError):
            self.engine.predict("X", 50, DRIVERS, context_mode="bogus")

    def test_load_default_engine(self):
        self.assertIsInstance(load_default_engine(), PredictionEngine)


class CompatibilityGuardTests(unittest.TestCase):
    def _dump(self, obj, name="m.joblib") -> Path:
        d = Path(tempfile.mkdtemp())
        p = d / name
        joblib.dump(obj, p)
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return p

    def test_wrong_feature_schema_raises(self):
        # A bundle declaring a different feature schema must be rejected at load.
        pipe = LogisticRegression().fit(np.array([[0], [1], [0], [1]]), [0, 1, 0, 1])
        path = self._dump({"pipeline": pipe, "feature_columns": ["A", "B"]})
        with self.assertRaises(IncompatibleModelError):
            PredictionEngine(path)

    def test_unrecognised_artifact_raises(self):
        path = self._dump({"not_a_pipeline": 123})
        with self.assertRaises(IncompatibleModelError):
            PredictionEngine(path)

    def test_missing_artifact_raises(self):
        with self.assertRaises(IncompatibleModelError):
            PredictionEngine(Path(tempfile.mkdtemp()) / "nope.joblib")

    def test_bare_pipeline_is_accepted(self):
        # A bare fitted estimator (candidate joblib style) with no declared schema is
        # accepted; the current FEATURE_COLUMNS are assumed and validated trivially.
        pipe = LogisticRegression().fit(np.array([[0], [1], [0], [1]]), [0, 1, 0, 1])
        path = self._dump(pipe)
        eng = PredictionEngine(path)
        self.assertEqual(eng.feature_columns, FEATURE_COLUMNS)
        self.assertEqual(eng.label_version_status, "unversioned_legacy")


if __name__ == "__main__":
    unittest.main()
