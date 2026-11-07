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

    def test_app_js_handles_osm_and_attribution(self):
        src = (Path(__file__).resolve().parents[1] / "web" / "assets" / "app.js").read_text(encoding="utf-8")
        self.assertIn("osm_centreline", src)
        self.assertIn("OpenStreetMap", src)
        # Mapping is explicit (event -> circuit), not a blind name guess.
        self.assertIn("event_to_circuit", src)

    def test_app_js_no_official_sector_claim(self):
        src = (Path(__file__).resolve().parents[1] / "web" / "assets" / "app.js").read_text(encoding="utf-8")
        # Must state official timing sectors are unavailable, and consume validated boundaries.
        self.assertIn("Official S1/S2/S3 timing sectors: not available", src)
        self.assertIn("marshal_boundaries", src)
        # The old equal-thirds visualization must be gone.
        self.assertNotIn("Math.floor(n / 3)", src)


@unittest.skipUnless((DATA / "index.json").is_file(), "intel data bundle not built")
class IntelMappingTests(unittest.TestCase):
    """Event->circuit mapping must be explicit and configuration-aware."""

    @classmethod
    def setUpClass(cls):
        cls.idx = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
        cls.c = _client()

    def test_index_exposes_mapping_and_coverage(self):
        m = self.c.get("/api/intel/index").json()
        self.assertIn("event_to_circuit", m["circuits"])
        self.assertIn("coverage", m)
        self.assertIn("events", m["coverage"])

    def test_barcelona_and_madrid_are_distinct_configurations(self):
        mapping = self.idx["circuits"]["event_to_circuit"]
        # Both exist in the 2026 calendar and must NOT resolve to the same circuit.
        if "Barcelona Grand Prix" in mapping and "Spanish Grand Prix" in mapping:
            self.assertNotEqual(mapping["Barcelona Grand Prix"], mapping["Spanish Grand Prix"],
                                "Barcelona GP and the Madrid 'Spanish GP' must be different circuits")

    def test_available_circuit_events_match_mapping(self):
        mapping = self.idx["circuits"]["event_to_circuit"]
        for c in self.idx["circuits"]["available"]:
            self.assertIn("geometry_type", c)
            for ev in c.get("events", []):
                self.assertEqual(mapping.get(ev), c["circuit_id"])

    def test_no_duplicate_circuit_ids(self):
        ids = [c["circuit_id"] for c in self.idx["circuits"]["available"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_osm_circuits_carry_attribution(self):
        for c in self.idx["circuits"]["available"]:
            if c.get("geometry_type") == "osm_centreline":
                d = self.c.get(f"/api/intel/circuit/{c['circuit_id']}").json()
                self.assertIn("openstreetmap", (d.get("licence") or "").lower())
                self.assertTrue(d.get("attribution"))
                self.assertEqual(d["corners"], [])          # OSM has no corner numbers

    def test_osm_loops_closed_by_topology_not_coord_stitching(self):
        # Every available OSM loop must be node-topology closed or a single closed way —
        # never bridged across distinct ways by coordinate proximity.
        for c in self.idx["circuits"]["available"]:
            if c.get("geometry_type") == "osm_centreline":
                d = self.c.get(f"/api/intel/circuit/{c['circuit_id']}").json()
                v = d["validation"]; joins = v.get("chain_joins") or {}
                self.assertEqual(joins.get("coord", 0), 0, f"{c['circuit_id']} relied on coordinate joins")
                self.assertTrue(v.get("closed_by_node_topology") or joins.get("node", 0) == 0,
                                f"{c['circuit_id']} multi-way loop not closed by node topology")
                self.assertIn("loop_gap_ratio", v)          # absolute + ratio both recorded

    def test_coverage_counts_consistent(self):
        cov = self.idx["coverage"]
        self.assertEqual(cov["verified"] + cov["partial"] + cov["unavailable"] + cov["failed"],
                         len(cov["circuits"]))
        # every circuit carries one of the four classifications
        valid = {"VERIFIED", "PARTIAL", "UNAVAILABLE", "FAILED"}
        for c in cov["circuits"]:
            self.assertIn(c["status"], valid)

    def test_verified_circuits_declare_official_sectors_unavailable(self):
        for c in self.idx["circuits"]["available"]:
            if c.get("geometry_type") == "fastlap_trace":
                d = self.c.get(f"/api/intel/circuit/{c['circuit_id']}").json()
                self.assertFalse(d["available"].get("official_timing_sectors", False))
                self.assertIn("official_timing_sectors", d)
                mp = d.get("marshal_projection") or {}
                self.assertTrue(mp.get("frame_compatible"))          # same coordinate frame
                self.assertIn("FastF1", mp.get("coordinate_reference", ""))

    def test_osm_circuits_have_no_cross_frame_marshal_markers(self):
        # OSM geometry must never carry FastF1 marshal markers projected across frames.
        for c in self.idx["circuits"]["available"]:
            if c.get("geometry_type") == "osm_centreline":
                d = self.c.get(f"/api/intel/circuit/{c['circuit_id']}").json()
                self.assertEqual(d.get("marshal_sectors", []), [])
                self.assertEqual(d.get("coordinate_system", "").count("FastF1"), 0)

    def test_build_manifest_present(self):
        cov = self.idx["coverage"]
        self.assertIn("build", cov)
        self.assertIn("build_id", cov["build"])


class GeometryValidationUnitTests(unittest.TestCase):
    """The builder's validation rejects malformed/invalid geometry."""

    def _validate(self):
        import importlib
        mod = importlib.import_module("degradiq.scripts.build_intel_data")
        return mod.validate_geometry

    def test_valid_geometry_passes(self):
        v = self._validate()
        good = {"outline": [[i, i] for i in range(0, 900, 20)], "corners": [{"number": 1, "xy": [50, 50]}]}
        self.assertEqual(v(good), [])

    def test_out_of_range_and_sparse_rejected(self):
        v = self._validate()
        self.assertTrue(v({"outline": [[0, 0], [1, 1]]}))                     # too few
        self.assertTrue(v({"outline": [[0, 0], [99999, 1]] * 30}))            # out of range
        self.assertTrue(v({"outline": [[float("nan"), 1]] * 30}))            # non-finite


def _square_outline(step=20):
    """A dense closed square loop in the 0-1000 normalised frame."""
    pts = []
    for x in range(100, 900, step): pts.append([x, 100])
    for y in range(100, 900, step): pts.append([900, y])
    for x in range(900, 100, -step): pts.append([x, 900])
    for y in range(900, 100, -step): pts.append([100, y])
    pts.append([100, 100])
    return pts


class MarshalProjectionUnitTests(unittest.TestCase):
    """Nearest-segment marshal projection with coordinate-frame + snap validation."""

    def _proj(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")._project_markers

    def test_missing_markers(self):
        r = self._proj()(_square_outline(), [])
        self.assertEqual(r["accepted"], []); self.assertEqual(r["boundaries"], [])

    def test_compatible_markers_accepted_small_snap(self):
        # markers sitting on the edges (same frame) -> accepted with tiny snap distance
        markers = [{"number": 1, "xy": [300, 100]}, {"number": 2, "xy": [900, 500]}, {"number": 3, "xy": [500, 900]}]
        r = self._proj()(_square_outline(), markers)
        self.assertEqual(r["summary"]["n_accepted"], 3)
        self.assertTrue(r["summary"]["max_snap_units"] < 5)
        self.assertTrue(r["summary"]["frame_compatible"])

    def test_excessive_snap_rejected(self):
        # a marker far inside the square (wrong frame / off-track) -> rejected, never forced on
        r = self._proj()(_square_outline(), [{"number": 9, "xy": [500, 500]}])
        self.assertEqual(r["summary"]["n_accepted"], 0)
        self.assertEqual(r["rejected"][0]["status"], "rejected")
        self.assertIn("snap distance", r["rejected"][0]["rejection_reason"])

    def test_markers_ordered_by_lap_distance(self):
        # given out of order, accepted markers come back sorted by cumulative distance
        markers = [{"number": 3, "xy": [500, 900]}, {"number": 1, "xy": [300, 100]}, {"number": 2, "xy": [900, 500]}]
        r = self._proj()(_square_outline(), markers)
        d = [m["cumulative_distance_units"] for m in r["accepted"]]
        self.assertEqual(d, sorted(d))

    def test_boundaries_are_outline_indices(self):
        markers = [{"number": 1, "xy": [300, 100]}, {"number": 2, "xy": [900, 500]}]
        r = self._proj()(_square_outline(), markers)
        n = len(_square_outline())
        self.assertTrue(all(0 <= b < n for b in r["boundaries"]))


class OSMRelationMemberTests(unittest.TestCase):
    """Relation member roles: pit-lane excluded, start node gives a point, start way does not."""

    def _candidates(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")._candidate_sources

    def _rel_data(self, with_start_node, with_start_way):
        # a route-type motorsport relation whose racing members (10,15,16) are highway=raceway
        # ways, so it passes positive validation; plus a pit way and optional start members.
        racing = [10, 15, 16]
        members = [{"type": "way", "ref": r, "role": "", "geometry": [{"lat": 0.0, "lon": float(i)}, {"lat": 0.001, "lon": float(i)}]}
                   for i, r in enumerate(racing)]
        members.append({"type": "way", "ref": 11, "role": "pit", "geometry": [{"lat": 0.0, "lon": 0.01}, {"lat": 0.0005, "lon": 0.01}]})
        if with_start_way:
            members.append({"type": "way", "ref": 12, "role": "start", "geometry": [{"lat": 0.0005, "lon": -0.0001}, {"lat": 0.0005, "lon": 0.0001}]})
        if with_start_node:
            members.append({"type": "node", "ref": 13, "role": "start", "lat": 0.0005, "lon": 0.0})
        els = [{"type": "relation", "id": 1, "tags": {"type": "route", "sport": "motor", "name": "Test Circuit"}, "members": members}]
        for wid in racing:  # standalone highway=raceway ways (positive racing evidence)
            els.append({"type": "way", "id": wid, "tags": {"highway": "raceway"}, "nodes": [wid, wid + 1],
                        "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.001, "lon": 0.0}]})
        return {"elements": els}

    def test_start_node_sets_point_and_pit_excluded(self):
        rels = [s for s in self._candidates()(self._rel_data(True, False), 2000) if s["kind"] == "relation"]
        src = rels[0]
        self.assertTrue(src["has_pit"])
        self.assertEqual(src["start"], (0.0005, 0.0))               # node -> defensible point
        self.assertEqual([w["id"] for w in src["ways"]], [10, 15, 16])  # pit + start excluded from the loop

    def test_start_way_recorded_but_no_point(self):
        rels = [s for s in self._candidates()(self._rel_data(False, True), 2000) if s["kind"] == "relation"]
        src = rels[0]
        self.assertTrue(src["start_way_present"])
        self.assertIsNone(src["start"])                             # never a point from an arbitrary way
        self.assertEqual([w["id"] for w in src["ways"]], [10, 15, 16])  # start way not stitched into loop


class NodeTopologyChainingTests(unittest.TestCase):
    """Ways chain by exact shared node ids; coordinate proximity never bridges unrelated ways."""

    def _chain(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")._components_with_ids

    def test_shared_node_connects(self):
        ways = [{"id": 1, "tags": {}, "nodes": [10, 11, 12], "pts": [(0, 0), (0, 1), (0, 2)]},
                {"id": 2, "tags": {}, "nodes": [12, 13, 14], "pts": [(0, 2), (0, 3), (0, 4)]}]
        comps = self._chain()(ways)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]["joins"]["node"], 1)
        self.assertEqual(comps[0]["joins"]["coord"], 0)

    def test_nearby_endpoints_different_nodes_do_not_connect(self):
        # identical endpoint COORDINATE but different node ids -> must stay separate
        ways = [{"id": 1, "tags": {}, "nodes": [10, 11, 12], "pts": [(0, 0), (0, 1), (0, 2.0)]},
                {"id": 2, "tags": {}, "nodes": [90, 91, 92], "pts": [(0, 2.0), (0, 3), (0, 4)]}]
        comps = self._chain()(ways)
        self.assertEqual(len(comps), 2)

    def test_coord_fallback_only_without_node_ids(self):
        ways = [{"id": 1, "tags": {}, "nodes": None, "pts": [(0, 0), (0, 1), (0, 2)]},
                {"id": 2, "tags": {}, "nodes": None, "pts": [(0, 2), (0, 3), (0, 4)]}]
        comps = self._chain()(ways)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]["joins"]["coord"], 1)

    def test_reversed_misordered_members_close_by_topology(self):
        # square loop nodes 1-2-3-4-1, members given reversed / out of order
        ways = [{"id": 1, "tags": {}, "nodes": [1, 2], "pts": [(0, 0), (1, 0)]},
                {"id": 2, "tags": {}, "nodes": [3, 2], "pts": [(1, 1), (1, 0)]},
                {"id": 3, "tags": {}, "nodes": [3, 4], "pts": [(1, 1), (0, 1)]},
                {"id": 4, "tags": {}, "nodes": [1, 4], "pts": [(0, 0), (0, 1)]}]
        comps = self._chain()(ways)
        self.assertEqual(len(comps), 1)
        self.assertTrue(comps[0]["closed_topology"])

    def test_single_closed_way_matching_nodes_is_topology_closed(self):
        way = [{"id": 1, "tags": {}, "nodes": [1, 2, 3, 1], "pts": [(0, 0), (1, 0), (1, 1), (0, 0)]}]
        c = self._chain()(way)[0]
        self.assertTrue(c["closed_topology"]); self.assertTrue(c["node_ids_available"])

    def test_single_way_distinct_end_nodes_nearby_coords_not_closed(self):
        # first/last node ids differ though the coordinates coincide -> NOT topology closed
        way = [{"id": 1, "tags": {}, "nodes": [1, 2, 3, 9], "pts": [(0, 0), (1, 0), (1, 1), (0, 0)]}]
        c = self._chain()(way)[0]
        self.assertFalse(c["closed_topology"]); self.assertTrue(c["node_ids_available"])

    def test_single_way_without_node_ids_flags_unavailable(self):
        way = [{"id": 1, "tags": {}, "nodes": None, "pts": [(0, 0), (1, 0), (1, 1), (0, 0)]}]
        c = self._chain()(way)[0]
        self.assertFalse(c["node_ids_available"]); self.assertFalse(c["closed_topology"])

    def test_coord_joined_multiway_is_not_topology_closed(self):
        # two coordinate-joined ways (no node ids) forming a loop must NOT claim topology closure
        ways = [{"id": 1, "tags": {}, "nodes": None, "pts": [(0, 0), (1, 0), (1, 1)]},
                {"id": 2, "tags": {}, "nodes": None, "pts": [(1, 1), (0, 1), (0, 0)]}]
        c = self._chain()(ways)[0]
        self.assertEqual(c["joins"]["coord"], 1)
        self.assertFalse(c["closed_topology"])           # coord joins can't be topology-closed
        self.assertFalse(c["node_ids_available"])


class RelationTypeFilterTests(unittest.TestCase):
    """Area/boundary relations must be rejected as racing-loop sources."""

    def _candidates(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")._candidate_sources

    def _data(self, rtype, roles, raceway_member_ids=(), rel_tags=None):
        # relation members (ways 10..) + optional standalone highway=raceway ways with those ids
        members = [{"type": "way", "ref": 10 + i, "role": r,
                    "geometry": [{"lat": 0.0, "lon": float(i)}, {"lat": 0.001, "lon": float(i)}]}
                   for i, r in enumerate(roles)]
        tags = {"type": rtype, "sport": "motor", "name": "X"}
        if rel_tags:
            tags.update(rel_tags)
        els = [{"type": "relation", "id": 1, "tags": tags, "members": members}]
        for wid in raceway_member_ids:
            els.append({"type": "way", "id": wid, "tags": {"highway": "raceway"}, "nodes": [wid * 10, wid * 10 + 1],
                        "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.001, "lon": 0.0}]})
        return {"elements": els}

    def test_multipolygon_inner_outer_rejected(self):
        srcs = self._candidates()(self._data("multipolygon", ["outer", "inner"]), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_sport_motor_with_inner_outer_rejected_even_without_type(self):
        srcs = self._candidates()(self._data(None, ["outer", "inner"]), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_non_multipolygon_without_racing_evidence_rejected(self):
        # a route relation whose members are NOT raceway ways -> no positive evidence -> rejected
        srcs = self._candidates()(self._data("route", ["", "forward"], raceway_member_ids=()), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_route_relation_with_raceway_members_accepted(self):
        # route + motorsport tag + members are highway=raceway ways (ids 10,11) -> accepted
        srcs = self._candidates()(self._data("route", ["", "forward"], raceway_member_ids=(10, 11)), 2000)
        rels = [s for s in srcs if s["kind"] == "relation"]
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["rel_type"], "route")
        ev = rels[0]["racing_evidence"]
        self.assertGreaterEqual(ev["raceway_by_count"]["fraction"], 0.8)
        self.assertGreaterEqual(ev["raceway_by_length"]["fraction"], 0.8)

    def test_bus_route_rejected(self):
        srcs = self._candidates()(self._data("route", ["", "forward"], raceway_member_ids=(10, 11),
                                             rel_tags={"route": "bus"}), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_generic_sport_motor_without_route_rejected(self):
        # type absent, sport=motor, but members are not raceway -> no positive evidence
        srcs = self._candidates()(self._data(None, ["", "forward"], raceway_member_ids=()), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_proposed_circuit_rejected_when_flag_off(self):
        import importlib
        m = importlib.import_module("degradiq.scripts.build_intel_data")
        self.assertFalse(m.ALLOW_PROPOSED_CIRCUIT_RELATIONS)
        srcs = self._candidates()(self._data("circuit", ["", "forward"], raceway_member_ids=(10, 11)), 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))

    def test_high_count_but_insufficient_length_rejected(self):
        # 2 of 2 eligible members are raceway by COUNT impossible to fail; craft 4 short raceway +
        # 1 long non-raceway -> count 0.8 but length < 0.8 -> rejected
        members = []
        for i in range(4):  # short raceway members (role "")
            members.append({"type": "way", "ref": 20 + i, "role": "",
                            "geometry": [{"lat": 0.0, "lon": float(i) * 0.001}, {"lat": 0.0, "lon": float(i) * 0.001 + 0.0005}]})
        members.append({"type": "way", "ref": 99, "role": "",   # long NON-raceway member
                        "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.05, "lon": 0.0}]})
        els = [{"type": "relation", "id": 1, "tags": {"type": "route", "sport": "motor", "name": "X"}, "members": members}]
        for wid in (20, 21, 22, 23):  # only the short ways are highway=raceway
            els.append({"type": "way", "id": wid, "tags": {"highway": "raceway"}, "nodes": [wid, wid + 1],
                        "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.0005}]})
        srcs = self._candidates()({"elements": els}, 2000)
        self.assertFalse(any(s["kind"] == "relation" for s in srcs))


class ClosureAndStartValidationTests(unittest.TestCase):
    """Tight OSM closure + metric start-node validation."""

    def _mod(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")

    def test_start_node_distance_projection(self):
        m = self._mod()
        loop = [(x, 0.0) for x in range(0, 1001, 50)] + [(1000.0, y) for y in range(0, 1001, 50)]
        on_loop = m._project_point_metric(loop, 500.0, 0.0)[0]
        self.assertLess(on_loop, 1.0)                   # a point on the loop snaps tight
        far = m._project_point_metric(loop, 500.0, 400.0)[0]
        self.assertGreater(far, m.START_SNAP_M)          # 400 m away exceeds the snap threshold


class PromotionRollbackTests(unittest.TestCase):
    """Failed post-promotion validation must restore the last known-good dataset."""

    def _mod(self):
        import importlib
        return importlib.import_module("degradiq.scripts.build_intel_data")

    def test_rollback_restores_backup(self):
        import tempfile, json as _json, shutil
        m = self._mod()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "circuits").mkdir(); (root / "_backup" / "circuits").mkdir(parents=True)
            good = {"circuit_id": "x", "outline": [[1, 1]] * 30, "geometry_type": "fastlap_trace"}
            # known-good backup
            (root / "_backup" / "circuits" / "x.json").write_text(_json.dumps(good))
            (root / "_backup" / "index.json").write_text('{"ok": true}')
            # a broken "promoted" state (x.json missing, index corrupted)
            (root / "index.json").write_text("CORRUPT")
            self.assertTrue(m._rollback(root))
            self.assertTrue((root / "circuits" / "x.json").is_file())      # restored
            self.assertEqual(_json.loads((root / "index.json").read_text())["ok"], True)

    def test_rollback_without_backup_returns_false(self):
        import tempfile
        m = self._mod()
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "circuits").mkdir()
            self.assertFalse(m._rollback(Path(td)))

    def test_validate_dataset_flags_missing_file(self):
        import tempfile, json as _json
        m = self._mod()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "circuits").mkdir()
            (root / "index.json").write_text(_json.dumps({
                "circuits": {"event_to_circuit": {"A Grand Prix": "a"}}, "coverage": {}}))
            probs = m._validate_dataset(root, [{"circuit_id": "a"}], {"A Grand Prix": "a"})
            self.assertTrue(any("file missing" in p for p in probs))

    def test_fault_injection_post_promotion_failure_rolls_back(self):
        # Full flow: a good promoted dataset + backup; inject a post-promotion failure (delete a
        # circuit file the index references); detect it; auto-rollback; revalidate clean.
        import tempfile, json as _json
        m = self._mod()
        good_geom = {"circuit_id": "a", "name": "A", "events": ["A Grand Prix"],
                     "outline": [[i, i] for i in range(0, 600, 20)], "corners": [], "geometry_type": "osm_centreline",
                     "source_dataset": "openstreetmap", "coordinate_reference": "x", "licence": "ODbL",
                     "osm": {"way_ids": [1]}}
        idx = {"circuits": {"event_to_circuit": {"A Grand Prix": "a"}}, "coverage": {}}
        cindex = [{"circuit_id": "a"}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sub in ("circuits", "_backup/circuits"):
                (root / sub).mkdir(parents=True)
            for base in (root, root / "_backup"):
                (base / "circuits" / "a.json").write_text(_json.dumps(good_geom))
                (base / "index.json").write_text(_json.dumps(idx))
            # inject failure: promoted circuit file goes missing
            (root / "circuits" / "a.json").unlink()
            self.assertTrue(m._validate_dataset(root, cindex, {"A Grand Prix": "a"}))   # detected
            self.assertTrue(m._rollback(root))                                          # auto-rollback
            self.assertEqual(m._validate_dataset(root, cindex, {"A Grand Prix": "a"}), [])  # clean again


if __name__ == "__main__":
    unittest.main()
