"""Regression tests: the API/engine response must not misrepresent the serving model.

The predict response previously hard-coded "logistic regression" in `model_type` while
a calibrated HGB model was served. These tests lock a model-independent description and
require the reported architecture to match the actually-served model.
"""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path

from degradiq.pit_strategy import MODEL_PATH
from degradiq.prediction_engine import load_default_engine

PROD = Path(MODEL_PATH).exists()
DRIVERS = [{"driver": "VER", "grid_position": 1, "starting_compound": "MEDIUM", "starting_tyre_age": 1}]


class ModelTypeSourceTests(unittest.TestCase):
    def test_model_type_is_model_independent_in_source(self):
        src = (Path(__file__).resolve().parents[1] / "pit_strategy.py").read_text(encoding="utf-8")
        # The response's model_type must not name a specific (and possibly wrong) algorithm.
        self.assertNotIn("logistic regression", src.lower())
        self.assertIn('"model_type": "discrete first-stop hazard model"', src)


@unittest.skipUnless(PROD, "no served model artifact")
class ResponseAccuracyTests(unittest.TestCase):
    def test_predict_response_matches_served_architecture(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
            from degradiq.api import app
        client = TestClient(app)
        r = client.post("/strategy/first-stop/predict", json={
            "event_name": "Australian Grand Prix", "race_laps": 58, "mode": "pre-race",
            "drivers": DRIVERS})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # Model-independent description, never a specific wrong algorithm.
        self.assertNotIn("logistic regression", body["model_type"].lower())
        # The engine block's architecture must equal the actually-served model.
        served = load_default_engine().provenance()["architecture"]
        self.assertEqual(body["engine"]["model"]["architecture"], served)
        self.assertEqual(body["engine"]["model"]["label_version_status"],
                         load_default_engine().provenance()["label_version_status"])


if __name__ == "__main__":
    unittest.main()
