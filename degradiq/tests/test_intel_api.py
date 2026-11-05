"""Tests for the Circuit & Tyre Intelligence read-only endpoints.

These assert real-data integrity: available resources carry provenance, unavailable
ones return an honest 404 (never a substituted circuit or fabricated tyre values),
and the id handling is safe against path traversal.
"""
from __future__ import annotations

import json
import unittest
import warnings
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "web" / "data"


def _client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
        from degradiq.api import app
    return TestClient(app)


class IntelIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _client()

    def test_index_ok(self):
        r = self.c.get("/api/intel/index")
        self.assertEqual(r.status_code, 200)
        m = r.json()
        self.assertEqual(m["status"], "ok")
        self.assertIn("available", m["circuits"])
        self.assertIn("events", m["tyre"])

    def test_meta_lists_intel(self):
        m = self.c.get("/api/meta").json()
        self.assertIn("intel_index", m)


@unittest.skipUnless((DATA / "index.json").is_file(), "intel data bundle not built")
class IntelCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _client()
        cls.idx = json.loads((DATA / "index.json").read_text(encoding="utf-8"))

    def test_available_circuit_has_geometry_and_provenance(self):
        avail = self.idx["circuits"]["available"]
        self.assertTrue(avail, "expected at least one verified circuit")
        cid = avail[0]["circuit_id"]
        d = self.c.get(f"/api/intel/circuit/{cid}").json()
        self.assertEqual(d["status"], "ok")
        self.assertGreater(len(d["outline"]), 20)          # real track path
        self.assertTrue(d["corners"])                      # numbered corners
        for key in ("source", "retrieved", "licence", "coordinate_system"):
            self.assertIn(key, d)                           # provenance present
        # Coordinates are finite numbers within the declared viewBox.
        for x, y in d["outline"][:50]:
            self.assertTrue(0 <= x <= 1000 and 0 <= y <= 1000)

    def test_unavailable_circuit_is_honest_404_not_substitute(self):
        r = self.c.get("/api/intel/circuit/nonexistent-circuit")
        self.assertEqual(r.status_code, 404)
        body = r.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertFalse(body["available"])
        self.assertNotIn("outline", body)                   # never a fake shape

    def test_id_validation_and_traversal_blocked(self):
        self.assertEqual(self.c.get("/api/intel/circuit/UPPER_bad").status_code, 422)
        # Encoded path traversal must not escape the data directory.
        self.assertIn(self.c.get("/api/intel/circuit/..%2f..%2fconfig").status_code, (404, 422))


@unittest.skipUnless((DATA / "index.json").is_file(), "intel data bundle not built")
class IntelTyreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _client()
        cls.idx = json.loads((DATA / "index.json").read_text(encoding="utf-8"))

    def test_available_tyre_has_real_observations_and_counts(self):
        events = self.idx["tyre"]["events"]
        self.assertTrue(events)
        eid = events[0]["event_id"]
        d = self.c.get(f"/api/intel/tyre/{eid}").json()
        self.assertEqual(d["status"], "ok")
        self.assertGreater(d["clean_laps"], 0)
        self.assertLessEqual(d["clean_laps"], d["total_laps"])
        self.assertIn("limitations", d)                     # honest caveats
        # Each compound carries sample counts and ordered age observations.
        for comp, cc in d["compounds"].items():
            self.assertGreater(cc["n"], 0)
            ages = [o["age"] for o in cc["observations"]]
            self.assertEqual(ages, sorted(ages))
            for o in cc["observations"]:
                self.assertGreaterEqual(o["n"], 1)          # real sample behind each point
                self.assertGreater(o["median_s"], 0)

    def test_unavailable_tyre_is_honest_404(self):
        r = self.c.get("/api/intel/tyre/nonexistent-event")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["status"], "unavailable")
        self.assertNotIn("compounds", r.json())


class IntelFrontendIntegrityTests(unittest.TestCase):
    """The client must surface unavailable states, never fabricate geometry/tyre values."""

    def test_app_js_has_honest_unavailable_and_no_random(self):
        src = (Path(__file__).resolve().parents[1] / "web" / "assets" / "app.js").read_text(encoding="utf-8")
        self.assertIn("renderCircuitUnavailable", src)
        self.assertIn("renderTyreUnavailable", src)
        self.assertIn("never substitute", src.lower())
        self.assertNotIn("Math.random", src)


if __name__ == "__main__":
    unittest.main()
