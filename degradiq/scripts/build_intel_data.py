"""Calendar-driven builder for Circuit + Tyre Intelligence data.

Produces version-controlled JSON under degradiq/web/data/ from two honest sources:

  1. FastF1 session telemetry (local cache) — a fastest-lap GPS *racing-line trace*
     plus real corner numbers / marshal sectors. Highest fidelity, but only for
     circuits whose position telemetry is cached (FastF1 live telemetry download is
     currently blocked upstream, so this cannot be expanded offline).
  2. OpenStreetMap, via the Overpass API — the circuit *centreline outline* from
     `highway=raceway` ways. Broad coverage for permanent circuits; no corner numbers
     or official sector boundaries. OSM data is © OpenStreetMap contributors, ODbL —
     redistributable with attribution.

Geometry is never invented: a circuit with neither cached telemetry nor a usable OSM
trace is recorded as unavailable with the reason, and the frontend shows an honest
state rather than a different circuit's shape. The event→circuit mapping is explicit
and configuration-aware (e.g. the 2026 Barcelona GP and the new Madrid "Spanish GP"
are different circuits and never share geometry).

Run locally, then commit degradiq/web/data/:

    python -m degradiq.scripts.build_intel_data            # circuits + tyre
    python -m degradiq.scripts.build_intel_data --circuits # geometry only (faster)
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.pit_strategy import PACKAGE_DIR  # noqa: E402

# FastF1 is only needed when actually building geometry/tyre data. Import it (and enable its
# cache) lazily so the module's pure helpers (projection, validation, OSM parsing) can be
# imported and unit-tested without FastF1 or its cache directory present (e.g. in CI).
_FF1 = None


def _fastf1():
    global _FF1
    if _FF1 is None:
        import fastf1 as _f
        (PACKAGE_DIR / "cache" / "fastf1").mkdir(parents=True, exist_ok=True)
        _f.Cache.enable_cache(str(PACKAGE_DIR / "cache" / "fastf1"))
        _FF1 = _f
    return _FF1


OUT = PACKAGE_DIR / "web" / "data"
NOW = datetime.now(timezone.utc).date().isoformat()

OSM_LICENCE = "© OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright)"
FF1_LICENCE = "Derived from FastF1-cached F1 timing/telemetry for research use; not for redistribution of raw F1 data."

# --------------------------------------------------------------------------------------
# Circuit registry — one entry per supported configuration. `events` lists the calendar
# EventName(s) that resolve to this configuration, so season/config changes never alias
# to the wrong layout. `fastf1` (year, event) uses cached telemetry when present; `osm`
# gives the Overpass search centre for a centreline trace otherwise.
# --------------------------------------------------------------------------------------
CIRCUITS = [
    {"id": "bahrain", "name": "Bahrain International Circuit", "location": "Sakhir", "country": "Bahrain",
     "events": ["Bahrain Grand Prix"], "fastf1": (2024, "Bahrain Grand Prix"), "osm": (26.0325, 50.5106)},
    {"id": "jeddah", "name": "Jeddah Corniche Circuit", "location": "Jeddah", "country": "Saudi Arabia",
     "events": ["Saudi Arabian Grand Prix"], "osm": (21.6319, 39.1044)},
    {"id": "albert-park", "name": "Albert Park Circuit", "location": "Melbourne", "country": "Australia",
     "events": ["Australian Grand Prix"], "osm": (-37.8497, 144.968)},
    {"id": "suzuka", "name": "Suzuka International Racing Course", "location": "Suzuka", "country": "Japan",
     "events": ["Japanese Grand Prix"], "osm": (34.8431, 136.5407)},
    {"id": "shanghai", "name": "Shanghai International Circuit", "location": "Shanghai", "country": "China",
     "events": ["Chinese Grand Prix"], "osm": (31.3389, 121.2197)},
    {"id": "miami", "name": "Miami International Autodrome", "location": "Miami Gardens", "country": "United States",
     "events": ["Miami Grand Prix"], "osm": (25.9581, -80.2389)},
    {"id": "imola", "name": "Autodromo Enzo e Dino Ferrari", "location": "Imola", "country": "Italy",
     "events": ["Emilia Romagna Grand Prix"], "osm": (44.3439, 11.7167)},
    {"id": "monaco", "name": "Circuit de Monaco", "location": "Monte Carlo", "country": "Monaco",
     "events": ["Monaco Grand Prix"], "osm": (43.7347, 7.4206),
     "config_note": "Street circuit — OSM raceway coverage is partial for public-road circuits."},
    {"id": "montreal", "name": "Circuit Gilles-Villeneuve", "location": "Montréal", "country": "Canada",
     "events": ["Canadian Grand Prix"], "osm": (45.5000, -73.5228)},
    {"id": "barcelona", "name": "Circuit de Barcelona-Catalunya", "location": "Montmeló", "country": "Spain",
     "events": ["Barcelona Grand Prix"], "fastf1": (2024, "Spanish Grand Prix"), "osm": (41.5700, 2.2611),
     "config_note": "The 2024 'Spanish Grand Prix' was held here at Barcelona-Catalunya; in 2026 it is the Barcelona GP."},
    {"id": "madring", "name": "Madring (Madrid Street Circuit)", "location": "Madrid", "country": "Spain",
     "events": ["Spanish Grand Prix"], "fastf1": (2026, "Spanish Grand Prix"), "osm": (40.4650, -3.6120),
     "config_note": "New-for-2026 Madrid circuit; hosted the 2026 Spanish GP (11-13 Sep 2026), construction completed "
                    "19 Oct 2026. Distinct circuit identity from Barcelona. Validated geometry may still be unavailable "
                    "if FastF1 telemetry is not accessible (403) and OSM lacks a validated raceway loop."},
    {"id": "red-bull-ring", "name": "Red Bull Ring", "location": "Spielberg", "country": "Austria",
     "events": ["Austrian Grand Prix"], "osm": (47.2197, 14.7647)},
    {"id": "silverstone", "name": "Silverstone Circuit", "location": "Silverstone", "country": "United Kingdom",
     "events": ["British Grand Prix"], "osm": (52.0733, -1.0147)},
    {"id": "spa", "name": "Circuit de Spa-Francorchamps", "location": "Spa", "country": "Belgium",
     "events": ["Belgian Grand Prix"], "osm": (50.4372, 5.9714)},
    {"id": "hungaroring", "name": "Hungaroring", "location": "Budapest", "country": "Hungary",
     "events": ["Hungarian Grand Prix"], "osm": (47.5789, 19.2486)},
    {"id": "zandvoort", "name": "Circuit Zandvoort", "location": "Zandvoort", "country": "Netherlands",
     "events": ["Dutch Grand Prix"], "osm": (52.3888, 4.5409)},
    {"id": "monza", "name": "Autodromo Nazionale Monza", "location": "Monza", "country": "Italy",
     "events": ["Italian Grand Prix"], "fastf1": (2024, "Italian Grand Prix"), "osm": (45.6156, 9.2811)},
    {"id": "baku", "name": "Baku City Circuit", "location": "Baku", "country": "Azerbaijan",
     "events": ["Azerbaijan Grand Prix"], "osm": (40.3725, 49.8533),
     "config_note": "Street circuit — OSM raceway coverage is partial for public-road circuits."},
    {"id": "marina-bay", "name": "Marina Bay Street Circuit", "location": "Marina Bay", "country": "Singapore",
     "events": ["Singapore Grand Prix"], "osm": (1.2914, 103.8640),
     "config_note": "Street circuit — OSM raceway coverage is partial for public-road circuits."},
    {"id": "cota", "name": "Circuit of the Americas", "location": "Austin", "country": "United States",
     "events": ["United States Grand Prix"], "osm": (30.1328, -97.6411)},
    {"id": "mexico", "name": "Autódromo Hermanos Rodríguez", "location": "Mexico City", "country": "Mexico",
     "events": ["Mexico City Grand Prix"], "osm": (19.4042, -99.0907)},
    {"id": "interlagos", "name": "Autódromo José Carlos Pace (Interlagos)", "location": "São Paulo", "country": "Brazil",
     "events": ["São Paulo Grand Prix"], "osm": (-23.7036, -46.6997)},
    {"id": "las-vegas", "name": "Las Vegas Strip Circuit", "location": "Las Vegas", "country": "United States",
     "events": ["Las Vegas Grand Prix"], "osm": (36.1147, -115.1730),
     "config_note": "Street circuit — OSM raceway coverage is partial for public-road circuits."},
    {"id": "lusail", "name": "Lusail International Circuit", "location": "Lusail", "country": "Qatar",
     "events": ["Qatar Grand Prix"], "osm": (25.4900, 51.4542)},
    {"id": "yas-marina", "name": "Yas Marina Circuit", "location": "Yas Island", "country": "United Arab Emirates",
     "events": ["Abu Dhabi Grand Prix"], "osm": (24.4672, 54.6031)},
]

# Tyre observations: 2024 cached races (unchanged — labelled by their historical season).
TYRE_EVENTS = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix",
    "Chinese Grand Prix", "Miami Grand Prix", "Emilia Romagna Grand Prix", "Monaco Grand Prix",
    "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Singapore Grand Prix", "United States Grand Prix", "Mexico City Grand Prix",
    "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]
TYRE_SEASON = 2024
COMPOUND_COLOR = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard", "INTERMEDIATE": "intermediate", "WET": "wet"}
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _tyre_slug(name: str) -> str:
    return name.lower().replace(" grand prix", "").strip().replace(" ", "-").replace("ã", "a").replace("ç", "c")


# ----------------------------------- geometry helpers ----------------------------------
def _normalise(points_xy, corners_xy=None):
    """Fit (x, y) metre coordinates into a 0-1000 viewBox with Y flipped for SVG."""
    P = np.asarray(points_xy, float)
    C = np.asarray(corners_xy, float) if corners_xy is not None and len(corners_xy) else P
    allx = np.concatenate([P[:, 0], C[:, 0]]); ally = np.concatenate([P[:, 1], C[:, 1]])
    minx, maxx, miny, maxy = allx.min(), allx.max(), ally.min(), ally.max()
    span = max(maxx - minx, maxy - miny) or 1.0

    def nm(x, y):
        return [round((x - minx) / span * 940 + 30, 1), round((maxy - y) / span * 940 + 30, 1)]
    params = {"minx": float(minx), "maxx": float(maxx), "miny": float(miny), "maxy": float(maxy),
              "span": float(span), "scale": 940.0 / span, "margin": 30.0, "y_flipped": True,
              "extent": (maxx - minx, maxy - miny)}
    return nm, params


MARSHAL_SNAP_THRESHOLD = 22.0  # normalised units (~2.2% of the 1000-unit viewBox)


def _project_markers(outline, markers, threshold=MARSHAL_SNAP_THRESHOLD):
    """Project marshal markers onto the nearest polyline SEGMENT of the outline.

    Operates in the shared, normalised FastF1 frame (outline + markers come from the same
    telemetry coordinate system). For each marker it finds the nearest point on a segment
    (not merely the nearest vertex), the cumulative lap distance, and the snap distance.
    Markers beyond the snap threshold, or near-equidistant to a distant segment (self-
    crossing / parallel straights), are REJECTED with a reason — never forced onto the loop.
    """
    if len(outline) < 4 or not markers:
        return {"accepted": [], "rejected": [], "boundaries": [],
                "summary": {"n_markers": len(markers), "n_accepted": 0, "n_rejected": len(markers),
                            "frame_compatible": True, "coordinate_reference": "FastF1 local XY (normalised)"}}
    cum = [0.0]
    for i in range(1, len(outline)):
        cum.append(cum[-1] + math.dist(outline[i - 1], outline[i]))
    total = cum[-1] or 1.0
    acc, rej = [], []
    for m in markers:
        px, py = m["xy"]; best = second = None
        for i in range(len(outline) - 1):
            ax, ay = outline[i]; bx, by = outline[i + 1]
            dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
            fx, fy = ax + t * dx, ay + t * dy
            dist = math.hypot(px - fx, py - fy)
            cand = (dist, i, [round(fx, 1), round(fy, 1)], cum[i] + t * math.hypot(dx, dy))
            if best is None or dist < best[0]:
                second = best; best = cand
            elif second is None or dist < second[0]:
                second = cand
        dist, i, foot, dalong = best
        rec = {"number": m.get("number"), "original_xy": m["xy"], "projected_xy": foot,
               "snap_distance_units": round(dist, 2), "cumulative_distance_units": round(dalong, 1), "segment_index": i}
        if dist > threshold:
            rec["status"] = "rejected"; rec["rejection_reason"] = f"snap distance {dist:.1f} exceeds threshold {threshold:.0f}"
            rej.append(rec); continue
        if second and second[0] < dist * 1.5 and abs(second[3] - dalong) > 0.08 * total and abs(second[1] - i) > 3:
            rec["status"] = "ambiguous"
            rec["rejection_reason"] = "near-equidistant to a distant segment (self-crossing / parallel straight); not resolved"
            rej.append(rec); continue
        rec["status"] = "accepted"; acc.append(rec)
    acc.sort(key=lambda r: r["cumulative_distance_units"])
    boundaries = sorted({min(range(len(cum)), key=lambda k: abs(cum[k] - r["cumulative_distance_units"])) for r in acc})
    snaps = [r["snap_distance_units"] for r in acc]
    summary = {"coordinate_reference": "FastF1 local XY (shared frame with outline & corners), normalised",
               "frame_compatible": True, "method": "nearest-segment projection with cumulative lap distance",
               "snap_threshold_units": round(threshold, 1),
               "max_snap_units": round(max(snaps), 2) if snaps else None,
               "mean_snap_units": round(sum(snaps) / len(snaps), 2) if snaps else None,
               "n_markers": len(markers), "n_accepted": len(acc), "n_rejected": len(rej)}
    return {"accepted": acc, "rejected": rej, "boundaries": boundaries, "summary": summary}


def build_circuit_fastf1(cfg) -> dict | None:
    fastf1 = _fastf1()
    year, event = cfg["fastf1"]
    s = fastf1.get_session(year, event, "R")
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    lap = s.laps.pick_fastest()
    tel = lap.get_telemetry()
    if "X" not in tel or len(tel) < 50:
        return None
    ci = s.get_circuit_info()
    rot = float(getattr(ci, "rotation", 0.0) or 0.0) / 180.0 * np.pi

    def rotate(x, y):
        return x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)

    X, Y = rotate(tel["X"].to_numpy(float), tel["Y"].to_numpy(float))
    cx, cy = rotate(ci.corners["X"].to_numpy(float), ci.corners["Y"].to_numpy(float))
    # ONE shared transform (rotation + normalisation) computed from the circuit telemetry
    # frame, then applied to outline, corners AND marshal markers — never normalised apart.
    nm, tparams = _normalise(np.column_stack([X, Y]), np.column_stack([cx, cy]))
    rotation_deg = round(float(getattr(ci, "rotation", 0.0) or 0.0), 3)
    step = max(1, len(X) // 360)
    outline = [nm(X[i], Y[i]) for i in range(0, len(X), step)]
    corners = [{"number": int(r["Number"]), "xy": nm(cx[i], cy[i])} for i, (_, r) in enumerate(ci.corners.iterrows())]
    raw_markers = []
    try:
        ms = ci.marshal_sectors
        if ms is not None and len(ms):
            raw_markers = [{"number": int(r["Number"]), "xy": nm(*rotate(float(r["X"]), float(r["Y"])))} for _, r in ms.iterrows()]
    except Exception:
        raw_markers = []
    # Project marshal markers (same FastF1 frame as the outline) with validation.
    proj = _project_markers(outline, raw_markers)
    marshal_ok = len(proj["accepted"]) > 0
    return {
        "geometry_type": "fastlap_trace", "geometry_label": "Fastest-lap GPS racing-line trace",
        "source_dataset": "fastf1", "geometry_source": "fastf1_telemetry",
        "source": "FastF1 session telemetry (fastest-lap track path + circuit_info corners/marshal sectors/rotation)",
        "source_event": f"{year} {event} · Race", "data_provider": "FastF1 / Formula 1 timing", "session": "Race",
        "licence": FF1_LICENCE,
        "coordinate_system": "normalised 0-1000 viewBox, circuit rotation applied, Y flipped for SVG",
        "coordinate_reference": "FastF1 local XY (metres), circuit rotation applied, normalised to 0-1000 viewBox",
        "construction_method": "fastest-lap telemetry path; corners/marshal sectors/rotation from FastF1 circuit_info",
        "coordinate_transform": {"rotation_deg": rotation_deg, "normalisation": tparams, "shared": True,
                                 "applies_to": ["outline", "corners", "marshal_markers"],
                                 "note": "one rotation + normalisation from the telemetry frame; annotation "
                                         "coordinates are never normalised independently"},
        "available": {
            "racing_loop": True, "outline": True, "corners": len(corners) > 0,
            "marshal_sectors": marshal_ok, "official_timing_sectors": False,
            "pit_lane": False, "start_finish": bool(outline),
        },
        "outline": outline, "corners": corners,
        "marshal_sectors": proj["accepted"], "marshal_rejected": proj["rejected"],
        "marshal_boundaries": proj["boundaries"], "marshal_projection": proj["summary"],
        "start_finish": outline[0] if outline else None, "n_outline_points": len(outline),
        "official_timing_sectors": "unavailable (FastF1 does not expose S1/S2/S3 boundary distances)",
        "notes": "Outline is the fastest lap's GPS path (approximate racing line, not a surveyed centreline). "
                 "Pit-lane geometry is not derived. Corner numbers and marshal sectors are from FastF1 circuit_info, in "
                 "the SAME coordinate frame as the outline. Shaded zones are the real FastF1 MARSHAL sectors "
                 "(nearest-segment projection, validated), NOT the three official timing sectors, whose boundary "
                 "distances are not available.",
    }


def _overpass(lat, lon, radius):
    """Fetch raceway ways AND motorsport circuit relations (with member geometry).

    highway=raceway is a strong signal but not the only OSM representation; many circuits
    are modelled as a relation whose member ways form the lap. We fetch both and only ever
    chain ways that belong together (standalone raceway ways, or one relation's members) —
    never unrelated ordinary roads.
    """
    # `out geom;` returns full way bodies — tags, the ordered node-id list AND the geometry —
    # so ways can be chained by exact shared node ids, not merely by coordinate proximity.
    q = (f'[out:json][timeout:50];('
         f'way["highway"="raceway"](around:{radius},{lat},{lon});'
         f'relation["highway"="raceway"](around:{radius},{lat},{lon});'
         f'relation["sport"="motor"](around:{radius},{lat},{lon});'
         f');out geom;')
    last = None
    for base in OVERPASS_MIRRORS:
        for attempt in range(2):
            try:
                req = urllib.request.Request(base, data=q.encode(),
                                             headers={"User-Agent": "DegradIQ-research/1.0 (circuit geometry)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r), q
            except Exception as e:  # 502s and timeouts are common on public mirrors
                last = e; time.sleep(2 + attempt * 2)
    raise last if last else RuntimeError("overpass failed")


def _path_metrics(path, lat0):
    mperlon = 111320 * math.cos(math.radians(lat0))
    xs = [p[1] * mperlon for p in path]; ys = [p[0] * 111320 for p in path]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    length = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(path)))
    gap = math.hypot(xs[0] - xs[-1], ys[0] - ys[-1])
    return span, length, gap


COORD_FALLBACK_TOL = 4.5e-5  # ~5 m — documented fallback ONLY where node topology is absent


def _near(a, b, tol=COORD_FALLBACK_TOL):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _components_with_ids(ways):
    """Chain ways into polylines by EXACT shared node ids (OSM topology).

    Two ways connect only when they share an endpoint node id. A small coordinate tolerance
    (~5 m) is used only when node ids are unavailable for a way (e.g. relation members from
    Overpass), and never to bridge two ways that both HAVE node ids but do not share one — so
    coordinate proximity can never silently connect unrelated ways. Tracks join provenance.
    """
    items = [w for w in ways if len(w["pts"]) >= 2]
    comps, used = [], set()

    def ep_nodes(w):
        n = w.get("nodes")
        return (n[0], n[-1]) if n and len(n) >= 2 else (None, None)

    def has_nodes(w):
        return bool(w.get("nodes") and len(w["nodes"]) >= 2)

    for i, seed in enumerate(items):
        if i in used:
            continue
        used.add(i)
        path = list(seed["pts"]); ids = [seed["id"]]; names = set(filter(None, [seed["tags"].get("name")]))
        hn, tn = ep_nodes(seed)
        node_ids_available = has_nodes(seed)
        joins = {"node": 0, "coord": 0}
        changed = True
        while changed:
            changed = False
            for j, w in enumerate(items):
                if j in used:
                    continue
                wn0, wn1 = ep_nodes(w); wp = w["pts"]
                attached = None  # ("tail"/"head", reversed, method)
                # 1) exact node-id connectivity
                if tn is not None:
                    if wn0 == tn:
                        attached = ("tail", False, "node")
                    elif wn1 == tn:
                        attached = ("tail", True, "node")
                if attached is None and hn is not None:
                    if wn1 == hn:
                        attached = ("head", False, "node")
                    elif wn0 == hn:
                        attached = ("head", True, "node")
                # 2) coordinate fallback ONLY when node topology is unavailable on a side
                if attached is None and (tn is None or wn0 is None or wn1 is None):
                    if _near(path[-1], wp[0]):
                        attached = ("tail", False, "coord")
                    elif _near(path[-1], wp[-1]):
                        attached = ("tail", True, "coord")
                    elif _near(path[0], wp[-1]):
                        attached = ("head", False, "coord")
                    elif _near(path[0], wp[0]):
                        attached = ("head", True, "coord")
                if attached is None:
                    continue
                side, rev, method = attached
                seg = wp[::-1] if rev else wp
                sn0, sn1 = (wn1, wn0) if rev else (wn0, wn1)
                if side == "tail":
                    path += seg[1:]; tn = sn1
                else:
                    path = seg[:-1] + path; hn = sn0
                ids.append(w["id"]); names |= set(filter(None, [w["tags"].get("name")]))
                node_ids_available = node_ids_available and has_nodes(w)
                joins[method] += 1; used.add(j); changed = True
        # Topology closure is only meaningful (and only claimed) when node ids are present AND
        # no join used coordinate proximity.
        closed_topology = node_ids_available and joins["coord"] == 0 and hn is not None and tn is not None and hn == tn
        comps.append({"pts": path, "ids": ids, "names": sorted(names), "joins": joins,
                      "node_ids_available": bool(node_ids_available), "closed_topology": bool(closed_topology)})
    return comps


EXPECTED_MIN_M, EXPECTED_MAX_M = 2800, 8000  # heuristic F1 circuit-length window
# Closure: a real circuit's chained raceway ways must meet end-to-end within a tight
# tolerance. We reject anything looser rather than let the SVG draw an artificial closing
# segment. Closure is established by node-id identity where OSM node refs exist; CLOSURE_ABS_M
# is the small ABSOLUTE coordinate tolerance used ONLY where node ids are genuinely unavailable
# (never 3% of the bounding span — proximity is not topology).
CLOSURE_ABS_M = 40.0
START_SNAP_M = 60.0  # a relation start node must lie within this of the racing-loop polyline


def _project_point_metric(pts, px, py):
    """Nearest point on a metric polyline to (px, py). Returns (distance_m, foot_xy, 2nd_dist)."""
    best = second = None
    for i in range(len(pts) - 1):
        ax, ay = pts[i]; bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        fx, fy = ax + t * dx, ay + t * dy
        d = math.hypot(px - fx, py - fy)
        if best is None or d < best[0]:
            second = best; best = (d, (fx, fy), i)
        elif second is None or d < second[0]:
            second = (d, (fx, fy), i)
    return best[0], best[1], (second[0] if second else None)


def _relation_racing_evidence(tags, member_way_ids, raceway_ids):
    """Positive racing-route test for a relation. Returns (ok, evidence_dict).

    A relation is a candidate racing route ONLY on positive evidence — not merely because it is
    'not a multipolygon'. We require a route-type relation with motorsport tags whose member ways
    are predominantly the fetched highway=raceway ways, OR a relation whose members are almost
    entirely raceway ways (strong geometry evidence). Area/boundary relations are excluded first.
    """
    rel_type = tags.get("type")
    n = len(member_way_ids)
    raceway_members = sum(1 for wid in member_way_ids if wid in raceway_ids)
    frac = raceway_members / n if n else 0.0
    is_route = rel_type in ("route", "route_master")
    motorsport_tag = tags.get("sport") == "motor" or tags.get("route") in ("motorsport", "race", "racing", "motor")
    ok = (is_route and motorsport_tag and frac >= 0.6) or frac >= 0.8
    evidence = {"relation_type": rel_type, "is_route_type": is_route, "motorsport_tag": bool(motorsport_tag),
                "member_ways": n, "raceway_member_ways": raceway_members, "raceway_member_fraction": round(frac, 2),
                "tags": {k: tags.get(k) for k in ("type", "sport", "route", "name", "highway", "leisure") if tags.get(k)}}
    return ok, evidence


def _candidate_sources(data, radius):
    """Build candidate way-sets: standalone highway=raceway ways, plus each relation's member ways
    that passes POSITIVE racing-route validation. Area/boundary relations and relations without
    racing evidence are rejected; unrelated roads are never stitched across sets."""
    standalone, relations_raw = [], []
    for el in data.get("elements", []):
        t = el.get("type"); tags = el.get("tags", {})
        if t == "way" and el.get("geometry"):
            if tags.get("area") == "yes" or tags.get("raceway") in ("pitlane", "paddock"):
                continue
            standalone.append({"id": el["id"], "tags": tags, "nodes": el.get("nodes"),
                               "pts": [(p["lat"], p["lon"]) for p in el["geometry"]]})
        elif t == "relation":
            relations_raw.append(el)
    raceway_ids = {w["id"] for w in standalone}

    rels = []
    for el in relations_raw:
        tags = el.get("tags", {}); rel_type = tags.get("type")
        member_roles = {(m.get("role") or "(none)") for m in el.get("members", [])}
        # 1) Hard reject AREA/boundary relations (multipolygon outer/inner rings are perimeters).
        if rel_type in ("multipolygon", "boundary") or ({"inner", "outer"} & member_roles):
            continue
        member_way_ids = [m.get("ref") for m in el.get("members", []) if m.get("type") == "way"]
        # 2) POSITIVE racing-route evidence required.
        ok, evidence = _relation_racing_evidence(tags, member_way_ids, raceway_ids)
        if not ok:
            continue
        mem, roles, start, has_pit, start_way = [], set(), None, False, False
        for m in el.get("members", []):
            role = m.get("role", "") or "(none)"
            roles.add(role)
            if m.get("type") == "way" and m.get("geometry"):
                if m.get("role") in ("pit", "pitlane", "pit_lane"):
                    has_pit = True; continue
                if m.get("role") in ("start", "finish", "start_finish"):
                    start_way = True; continue
                mem.append({"id": m.get("ref"), "tags": {"name": tags.get("name")}, "nodes": m.get("nodes"),
                            "pts": [(p["lat"], p["lon"]) for p in m["geometry"]], "role": m.get("role")})
            elif m.get("type") == "node" and m.get("lat") is not None and "start" in (m.get("role") or ""):
                start = (m["lat"], m["lon"])
        if mem:
            rels.append({"kind": "relation", "ways": mem, "rel_id": el["id"], "rel_name": tags.get("name"),
                         "rel_type": rel_type, "rel_tags": evidence["tags"], "racing_evidence": evidence,
                         "roles": sorted(roles), "has_pit": has_pit, "start": start,
                         "start_way_present": start_way, "n_members": len(el.get("members", []))})
    sources = list(rels)
    if standalone:
        sources.append({"kind": "raceway_ways", "ways": standalone, "rel_id": None, "rel_name": None})
    return sources


def build_circuit_osm(cfg) -> tuple[dict | None, str | None]:
    """Return (geometry, None) on a validated circuit loop, else (None, reason).

    Validation records WHY a candidate is accepted or rejected. The length window is a
    heuristic: a named circuit relation forming a clean closed loop can be accepted just
    outside it, with the reason recorded.
    """
    lat, lon = cfg["osm"]
    radius = cfg.get("osm_radius", 2800)
    data, query = _overpass(lat, lon, radius)
    sources = _candidate_sources(data, radius)
    if not sources:
        return None, "no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit)"

    rejects, best, best_score = [], None, -1e9
    for src in sources:
        for comp in _components_with_ids(src["ways"]):
            path = comp["pts"]
            if len(path) < 25:
                continue
            span, length, gap = _path_metrics(path, lat)
            topo_closed = comp.get("closed_topology")      # node-id closed AND zero coord joins
            nj, cj = comp["joins"]["node"], comp["joins"]["coord"]
            nia = comp.get("node_ids_available")
            # Closure basis: where node ids exist, require exact first/last node-id identity (and
            # no coordinate joins). Only where node ids are genuinely unavailable do we fall back
            # to a small ABSOLUTE coordinate tolerance — never 3% of the bounding span.
            if nia:
                geom_ok = topo_closed
                closure_basis = "node-topology: first/last node ids match, zero coordinate joins" if geom_ok else None
            else:
                geom_ok = gap <= CLOSURE_ABS_M
                closure_basis = f"absolute coordinate tolerance {round(CLOSURE_ABS_M)}m (OSM node ids unavailable)" if geom_ok else None
            in_range = EXPECTED_MIN_M <= length <= EXPECTED_MAX_M
            named = bool(comp["names"]) or bool(src.get("rel_name"))
            rel = src["kind"] == "relation"
            accept = geom_ok and (in_range or (rel and named and 2000 <= length <= 9000))
            if not accept:
                if not geom_ok:
                    why = (f"not node-topology closed (node joins {nj}, coord joins {cj})" if nia
                           else f"closure gap {round(gap)}m exceeds absolute tolerance {round(CLOSURE_ABS_M)}m")
                else:
                    why = "length %dm outside expected window" % length
                rejects.append(f"{src['kind']} loop ({round(length)}m, gap {round(gap)}m): {why}")
                continue
            # Prefer node-topology closed loops and node joins; penalise any coordinate joins.
            score = length - gap * 4 + (1500 if topo_closed else 0) + (800 if rel else 0) \
                + (400 if named else 0) + 20 * nj - 80 * cj
            if score > best_score:
                best = (comp, src, span, length, gap, named, in_range, closure_basis); best_score = score
    if best is None:
        detail = "; ".join(rejects[:4]) if rejects else "no candidate loops"
        return None, f"no validated closed circuit loop in OpenStreetMap ({detail})"

    comp, src, span, length, gap, named, in_range, closure_basis = best
    accepted_reason = (closure_basis or "") + ("" if in_range else
                      f" · length {round(length)}m accepted on validated circuit-route identity")
    mperlon = 111320 * math.cos(math.radians(lat))
    pts = [(p[1] * mperlon, p[0] * 111320) for p in comp["pts"]]
    stepn = max(1, len(pts) // 360)
    pts = pts[::stepn]
    nm, _ = _normalise(pts)
    outline = [nm(x, y) for x, y in pts]
    ids = [i for i in comp["ids"] if i is not None][:40]
    way_urls = [f"https://www.openstreetmap.org/way/{i}" for i in ids[:20]]
    rel_url = [f"https://www.openstreetmap.org/relation/{src['rel_id']}"] if src.get("rel_id") else []
    # A relation is treated as candidate discovery, not proof of identity: it is accepted only
    # because its member ways chain into a validated closed loop near the known circuit centre.
    rel_validation = None
    start_finish = None
    start_validation = None
    pit_lane_present = False
    if src["kind"] == "relation":
        start = src.get("start")
        if start:
            # Validate the start NODE against the selected racing-loop polyline: project it in
            # metric coordinates, require it within START_SNAP_M and not ambiguous; otherwise
            # reject it rather than render an unvalidated node as the start/finish.
            sx, sy = start[1] * mperlon, start[0] * 111320
            sdist, foot, second = _project_point_metric(pts, sx, sy)
            ambiguous = second is not None and second < sdist * 1.4 and second < START_SNAP_M
            if sdist <= START_SNAP_M and not ambiguous:
                start_finish = nm(foot[0], foot[1])
                start_validation = {"status": "accepted", "snap_distance_m": round(sdist, 1),
                                    "threshold_m": START_SNAP_M, "projected_onto_loop": True}
            else:
                start_validation = {"status": "rejected", "snap_distance_m": round(sdist, 1),
                                    "threshold_m": START_SNAP_M,
                                    "reason": "ambiguous placement" if ambiguous else "start node too far from racing loop"}
        pit_lane_present = bool(src.get("has_pit"))
        rel_validation = {"relation_id": src.get("rel_id"), "relation_name": src.get("rel_name"),
                          "relation_type": src.get("rel_type"), "relation_tags": src.get("rel_tags"),
                          "racing_route_evidence": src.get("racing_evidence"),
                          "member_ways_used": len(comp["ids"]), "member_roles": src.get("roles"),
                          "pit_lane_member_present": pit_lane_present,
                          "start_node_member_present": bool(start), "start_way_member_present": bool(src.get("start_way_present")),
                          "start_node_validation": start_validation,
                          "start_finish_source": ("relation start node (validated)" if start_finish else
                                                  ("relation start node rejected" if start else
                                                   ("start way present but not point-mapped" if src.get("start_way_present") else "none"))),
                          "connected_closed_loop": True,
                          "identity_basis": "accepted on POSITIVE racing-route evidence (route-type + motorsport tags "
                                            "and/or members predominantly highway=raceway) AND a node-topology closed "
                                            "loop at the known circuit centre; a non-multipolygon tag alone is not proof"}
    geom = {
        "geometry_type": "osm_centreline", "geometry_label": "OpenStreetMap circuit centreline (approximate outline)",
        "source_dataset": "openstreetmap", "geometry_source": "openstreetmap_overpass",
        "coordinate_reference": "WGS84 lat/lon → local equirectangular metres → normalised 0-1000 viewBox",
        "construction_method": "validated closed raceway loop chained from OSM ways / relation members",
        "source": f"OpenStreetMap {'relation + ' if src['kind'] == 'relation' else ''}highway=raceway ways near {lat:.4f},{lon:.4f} (Overpass API)",
        "source_event": None, "data_provider": "OpenStreetMap contributors", "session": None,
        "licence": OSM_LICENCE, "attribution": "© OpenStreetMap contributors",
        "licence_compliance": ("ODbL 1.0: these outlines are a Derivative Database of OpenStreetMap, published under "
                               "ODbL 1.0 (share-alike). Attribution alone is NOT sufficient. The source can be "
                               "reconstructed from the recorded OSM way/relation ids and Overpass query. "
                               "See degradiq/web/data/OSM_ATTRIBUTION.md."),
        "osm": {"way_ids": ids, "relation_id": src.get("rel_id"), "names": comp["names"] or ([src["rel_name"]] if src.get("rel_name") else []),
                "source_urls": rel_url + way_urls, "query": query,
                "pit_lane_in_osm": pit_lane_present, "relation_validation": rel_validation,
                "retrieved_utc": datetime.now(timezone.utc).isoformat()},
        "validation": {"length_m": round(length), "span_m": round(span), "loop_gap_m": round(gap),
                       "loop_gap_ratio": round(gap / max(span, 1), 4),
                       "loop_closed": True, "closed_by_node_topology": bool(comp.get("closed_topology")),
                       "node_ids_available": bool(comp.get("node_ids_available")),
                       "closure_basis": closure_basis, "closure_abs_tolerance_m": round(CLOSURE_ABS_M),
                       "chain_joins": comp.get("joins"),
                       "expected_window_m": [EXPECTED_MIN_M, EXPECTED_MAX_M],
                       "within_expected_window": bool(in_range), "named": bool(named),
                       "from_relation": src["kind"] == "relation", "relation_type": src.get("rel_type"),
                       "accepted_reason": accepted_reason, "start_node": start_validation},
        "coordinate_system": "normalised 0-1000 viewBox (equirectangular projection), Y flipped for SVG",
        "available": {"outline": True, "corners": False, "marshal_sectors": False, "pit_lane": False,
                      "start_finish": start_finish is not None, "pit_lane_known_in_osm": pit_lane_present},
        "missing": [m for m in ["corner numbers", "sector boundaries", "pit lane", "racing direction", "start/finish line"]
                    if not (m == "start/finish line" and start_finish)],
        "outline": outline, "corners": [], "marshal_sectors": [],
        "start_finish": start_finish, "n_outline_points": len(outline),
        "track_length_m": round(length), "loop_closed": True, "loop_gap_m": round(gap),
        "notes": "Approximate circuit centreline from OpenStreetMap (© OpenStreetMap contributors, ODbL). It is a "
                 "validated CLOSED raceway loop — a surveyed outline, NOT a racing line and NOT official timing "
                 "geometry. Corner numbers, sector boundaries, pit lane, start/finish and racing direction are not "
                 "available from this source and are not shown.",
    }
    return geom, None


def schema_ok(d) -> list[str]:
    """Current-schema requirements for a known-good circuit file (used before carry-forward)."""
    probs = validate_geometry(d)
    if d.get("geometry_type") not in ("fastlap_trace", "osm_centreline"):
        probs.append("missing/invalid geometry_type")
    for key in ("source_dataset", "coordinate_reference", "licence"):
        if not d.get(key):
            probs.append(f"missing {key} (pre-current-schema artifact)")
    if d.get("geometry_type") == "osm_centreline" and not (d.get("osm") or {}).get("way_ids"):
        probs.append("OSM file missing way-id provenance")
    return probs


def validate_geometry(d) -> list[str]:
    """Return a list of problems; empty means the geometry is usable."""
    problems = []
    ol = d.get("outline") or []
    if len(ol) < 20:
        problems.append(f"too few outline points ({len(ol)})")
    bad = [p for p in ol if not (isinstance(p, list) and len(p) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) and -10 <= v <= 1010 for v in p))]
    if bad:
        problems.append(f"{len(bad)} invalid coordinates")
    for c in d.get("corners") or []:
        xy = c.get("xy")
        if not (xy and len(xy) == 2 and all(math.isfinite(v) for v in xy)):
            problems.append("invalid corner coordinate"); break
    return problems


def build_one_circuit(cfg) -> tuple[dict | None, dict]:
    """Return (geometry_or_None, status_record).

    Classifications: VERIFIED (FastF1 racing-line + corners), PARTIAL (validated OSM
    closed-loop outline, no corners), UNAVAILABLE (no usable source), FAILED (a source
    was attempted but acquisition or validation failed).
    """
    base = {"circuit_id": cfg["id"], "name": cfg["name"], "location": cfg.get("location"),
            "country": cfg.get("country"), "retrieved": NOW, "config_note": cfg.get("config_note")}
    ff1_note = None
    # 1) FastF1 cached telemetry (richest). A controlled single attempt — a 403/not-cached
    #    error does NOT fail the circuit; it falls through to OSM (respecting the 403 limit).
    if cfg.get("fastf1"):
        try:
            g = build_circuit_fastf1(cfg)
            if g:
                g.update(base)
                probs = validate_geometry(g)
                # VERIFIED requires more than a closed shape: real corner numbers, a resolved
                # event mapping and valid coordinates — i.e. a validated circuit identity.
                if not g.get("corners"):
                    probs.append("no corner data (not a verified circuit identity)")
                if not cfg.get("events"):
                    probs.append("circuit has no calendar event mapping")
                if not probs:
                    g["identity_validated"] = True
                    g["identity_basis"] = ("FastF1 circuit_info corners + fixed event→circuit mapping "
                                           f"({cfg['fastf1'][0]} {cfg['fastf1'][1]} → {cfg['id']})")
                    return g, {"status": "VERIFIED", "geometry_type": "fastlap_trace", "has_corners": True,
                               "source_dataset": "fastf1", "corners": len(g["corners"])}
                ff1_note = "FastF1 geometry invalid: " + "; ".join(probs)
            else:
                ff1_note = "FastF1 returned no telemetry"
        except Exception as e:
            ff1_note = f"FastF1 not accessible ({type(e).__name__}; e.g. 403/not cached)"
        if ff1_note and not cfg.get("osm"):
            return None, {"status": "FAILED", "reason": ff1_note, "attempted": "fastf1"}
    # 2) OpenStreetMap centreline
    if cfg.get("osm"):
        try:
            g, reason = build_circuit_osm(cfg)
            if g:
                g.update(base)
                probs = validate_geometry(g)
                if not probs:
                    return g, {"status": "PARTIAL", "geometry_type": "osm_centreline", "has_corners": False,
                               "source_dataset": "openstreetmap", "track_length_m": g.get("track_length_m"),
                               "osm_way_ids": g["osm"]["way_ids"][:8], "osm_relation_id": g["osm"].get("relation_id"),
                               "missing": g.get("missing")}
                return None, {"status": "FAILED", "reason": "OSM geometry invalid: " + "; ".join(probs), "attempted": "osm"}
            full = (ff1_note + "; " if ff1_note else "") + reason
            return None, {"status": "UNAVAILABLE", "reason": full, "attempted": ("fastf1+osm" if ff1_note else "osm")}
        except Exception as e:
            return None, {"status": "FAILED", "reason": f"OSM acquisition failed: {type(e).__name__}", "attempted": "osm"}
    return None, {"status": "UNAVAILABLE", "reason": "no data source configured"}


# ----------------------------------- tyre (unchanged) ----------------------------------
def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()
    keep = df["LapTime_s"].notna() & df["Compound"].notna() & df["TyreLife"].notna()
    keep &= df["PitInTime"].isna() & df["PitOutTime"].isna()
    if "IsAccurate" in df.columns:
        keep &= df["IsAccurate"].fillna(False)
    if "TrackStatus" in df.columns:
        ts = df["TrackStatus"].astype(str)
        keep &= ~ts.str.contains("4") & ~ts.str.contains("5") & ~ts.str.contains("6") & ~ts.str.contains("7")
    return df[keep]


def build_tyre(year: int, name: str) -> dict | None:
    fastf1 = _fastf1()
    s = fastf1.get_session(year, name, "R")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    total = len(s.laps)
    clean = _clean_laps(s.laps)
    if clean.empty:
        return None
    compounds = {}
    for comp, g in clean.groupby("Compound"):
        comp = str(comp).upper()
        if comp not in COMPOUND_COLOR:
            continue
        rows = []
        for age, gg in g.groupby(g["TyreLife"].astype(int)):
            lt = gg["LapTime_s"].to_numpy()
            med = float(np.median(lt)); lt = lt[lt <= med * 1.07]
            if len(lt) < 3:
                continue
            rows.append({"age": int(age), "median_s": round(float(np.median(lt)), 3),
                         "p25_s": round(float(np.percentile(lt, 25)), 3), "p75_s": round(float(np.percentile(lt, 75)), 3),
                         "n": int(len(lt))})
        if not rows:
            continue
        base = rows[0]["median_s"]
        for r in rows:
            r["pace_loss_s"] = round(r["median_s"] - base, 3)
        compounds[comp] = {"color": COMPOUND_COLOR[comp], "n": int(len(g)),
                           "age_range": [rows[0]["age"], rows[-1]["age"]], "observations": rows}
    if not compounds:
        return None
    return {
        "event_id": _tyre_slug(name), "event": name, "season": year, "session": "Race",
        "source": "FastF1-cached race lap timing", "retrieved": NOW,
        "metric": "actual lap time (seconds), no fuel/traffic correction — raw observations",
        "cleaning": "excluded in/out laps, IsAccurate==False, and laps under Safety-Car/VSC/red-flag track status; "
                    "per tyre-age group dropped lap times > 107% of the group median.",
        "limitations": "Lap time is affected by fuel load, traffic, track evolution and driver — not tyre age alone. "
                       "Pace-loss is relative to the youngest clean age shown, not a validated degradation model. "
                       "No tyre temperature/pressure/wear is measured.",
        "total_laps": int(total), "clean_laps": int(len(clean)), "compounds": compounds,
    }


# ----------------------------------------- main ----------------------------------------
def load_calendar():
    p = PACKAGE_DIR / "calendar_2026.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


def main(circuits_only=False):
    import os, shutil, uuid
    (OUT / "circuits").mkdir(parents=True, exist_ok=True)
    (OUT / "tyre").mkdir(parents=True, exist_ok=True)
    calendar = load_calendar()
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    build_utc = datetime.now(timezone.utc).isoformat()

    event_to_circuit = {ev: cfg["id"] for cfg in CIRCUITS for ev in cfg["events"]}

    # --- Phase 1: build everything into a run-specific STAGING directory ---
    STAGE = OUT / "_staging"
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "circuits").mkdir(parents=True)

    source_failures, circuits_index, coverage = [], [], []
    for cfg in CIRCUITS:
        geom, status = build_one_circuit(cfg)
        rec = {"circuit_id": cfg["id"], "name": cfg["name"], "events": cfg["events"], **status}
        if status["status"] == "FAILED":
            source_failures.append({"circuit_id": cfg["id"], "reason": status.get("reason")})
        if geom:
            (STAGE / "circuits" / f"{cfg['id']}.json").write_text(json.dumps(geom), encoding="utf-8")
            circuits_index.append({
                "circuit_id": cfg["id"], "name": cfg["name"], "events": cfg["events"], "available": True,
                "geometry_type": geom["geometry_type"], "source_dataset": geom.get("source_dataset"),
                "geometry_source": geom.get("geometry_source"), "has_corners": geom["available"].get("corners", False),
                "availability": geom.get("available"), "config_note": cfg.get("config_note"),
            })
        coverage.append(rec)
        print(f"[circuit] {cfg['id']:14} {status['status']:11} "
              f"{status.get('geometry_type', status.get('reason', ''))}", flush=True)

    events_cov = []
    for e in calendar:
        name = e.get("EventName"); cid = event_to_circuit.get(name)
        cover = next((c for c in coverage if c["circuit_id"] == cid), None)
        events_cov.append({"round": e.get("RoundNumber"), "event": name, "circuit_id": cid,
                           "status": (cover or {}).get("status", "unmapped"),
                           "geometry_type": (cover or {}).get("geometry_type")})

    tyre = []
    if not circuits_only:
        for name in TYRE_EVENTS:
            try:
                t = build_tyre(TYRE_SEASON, name)
                if t:
                    (OUT / "tyre" / f"{t['event_id']}.json").write_text(json.dumps(t), encoding="utf-8")
                    tyre.append({"event_id": t["event_id"], "event": name, "season": TYRE_SEASON,
                                 "compounds": list(t["compounds"].keys()), "clean_laps": t["clean_laps"]})
                    print(f"[tyre] {name}: {t['clean_laps']} clean laps", flush=True)
            except Exception as e:
                print(f"[tyre] {name}: FAILED {type(e).__name__}", flush=True)

    counts = {k: sum(1 for c in coverage if c["status"] == k) for k in ("VERIFIED", "PARTIAL", "UNAVAILABLE", "FAILED")}

    def build_index(cindex, cov, served_prev):
        return {
            "generated": NOW, "build_id": build_id, "build_utc": build_utc,
            "sources": {"fastf1": "FastF1 session telemetry (cached)", "osm": "OpenStreetMap via Overpass API (ODbL)"},
            "circuits": {"available": cindex, "event_to_circuit": event_to_circuit,
                         "note": "Geometry is FastF1 racing-line (with corners) or a validated OpenStreetMap closed-loop "
                                 "centreline (outline only); circuits with neither are unavailable and never substituted."},
            "coverage": {"supported_events": len(calendar), "circuits_total": len(CIRCUITS),
                         "verified": counts["VERIFIED"], "partial": counts["PARTIAL"],
                         "unavailable": counts["UNAVAILABLE"], "failed": counts["FAILED"],
                         "classifications": ["VERIFIED", "PARTIAL", "UNAVAILABLE", "FAILED"],
                         "build": {"build_id": build_id, "build_utc": build_utc,
                                   "source_failures": source_failures, "served_from_previous": served_prev},
                         "circuits": cov, "events": events_cov},
            "tyre": {"season": TYRE_SEASON, "events": tyre} if not circuits_only else _keep_existing_tyre(),
        }

    (STAGE / "index.json").write_text(json.dumps(build_index(circuits_index, coverage, []), indent=2), encoding="utf-8")

    # --- Phase 2: validate STAGING before promotion ---
    problems = _validate_dataset(STAGE, circuits_index, event_to_circuit)
    verified_ids = {c["circuit_id"] for c in coverage if c["status"] == "VERIFIED"}
    missing_verified = [cid for cid in verified_ids if not (STAGE / "circuits" / f"{cid}.json").is_file()]
    if problems or missing_verified:
        print("\nSTAGING VALIDATION FAILED — NOT promoting; existing known-good dataset preserved.", flush=True)
        for p in problems + [f"missing VERIFIED circuit {m}" for m in missing_verified]:
            print("  -", p, flush=True)
        print(f"Staging left at {STAGE} for inspection. No files in {OUT/'circuits'} were changed.", flush=True)
        return

    # --- Phase 3: promote (backup for rollback; preserve known-good on acquisition FAILED) ---
    served_prev = _promote(STAGE, circuits_index, coverage, source_failures, os, shutil)
    index = build_index(circuits_index, coverage, served_prev)
    tmp = OUT / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    os.replace(tmp, OUT / "index.json")
    shutil.rmtree(STAGE, ignore_errors=True)
    _write_osm_attribution(circuits_index)
    _write_coverage_report(index)
    post = _validate_dataset(OUT, circuits_index, event_to_circuit)
    print(f"\nBuild {build_id}: {counts['VERIFIED']} VERIFIED, {counts['PARTIAL']} PARTIAL, "
          f"{counts['UNAVAILABLE']} UNAVAILABLE, {counts['FAILED']} FAILED (of {len(CIRCUITS)}); "
          f"available={len(circuits_index)}; served_from_previous={len(served_prev)}; events={len(calendar)}", flush=True)
    if post:
        print("POST-PROMOTION VALIDATION FAILED — automatically rolling back to last known-good dataset.", flush=True)
        for p in post:
            print("  -", p, flush=True)
        restored = _rollback(OUT)
        print("ROLLED BACK." if restored else "ROLLBACK: no backup available (dataset left as-is for inspection).", flush=True)
        return
    print("PROMOTED. Post-promotion validation: OK", flush=True)
    print(f"Rollback copy at {OUT/'_backup'}.", flush=True)


def _rollback(root) -> bool:
    """Restore the last known-good dataset from <root>/_backup. Returns True if restored."""
    import shutil
    backup = root / "_backup"
    if not (backup / "index.json").is_file() or not (backup / "circuits").exists():
        return False
    for f in (root / "circuits").glob("*.json"):
        f.unlink()
    for f in (backup / "circuits").glob("*.json"):
        shutil.copy2(f, root / "circuits" / f.name)
    shutil.copy2(backup / "index.json", root / "index.json")
    return True


def _promote(STAGE, circuits_index, coverage, source_failures, os, shutil):
    """Promote staging into OUT. Back up current data for rollback; carry forward a previous
    known-good circuit file when THIS run's acquisition FAILED (do not prune on provider error)."""
    backup = OUT / "_backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)
    if (OUT / "circuits").exists():
        shutil.copytree(OUT / "circuits", backup / "circuits")
    if (OUT / "index.json").is_file():
        shutil.copy2(OUT / "index.json", backup / "index.json")

    staged_ids = {c["circuit_id"] for c in circuits_index}
    failed_ids = {f["circuit_id"] for f in source_failures}
    served_prev = []
    # Carry forward previous good files for circuits that FAILED acquisition this run.
    for cid in failed_ids:
        if cid in staged_ids:
            continue
        prev = OUT / "circuits" / f"{cid}.json"
        if prev.is_file():
            try:
                d = json.loads(prev.read_text(encoding="utf-8"))
                # Never blindly carry forward: revalidate against the CURRENT schema, geometry
                # and provenance rules before preserving as known-good.
                sp = schema_ok(d)
                if sp:
                    print(f"[carry] dropped stale {cid} (fails current schema: {'; '.join(sp[:2])})", flush=True)
                    continue
                circuits_index.append({
                    "circuit_id": cid, "name": d.get("name"), "events": d.get("events") or [],
                    "available": True, "geometry_type": d.get("geometry_type"),
                    "source_dataset": d.get("source_dataset"), "has_corners": (d.get("available") or {}).get("corners", False),
                    "availability": d.get("available"), "served_from_previous_build": True,
                })
                for rec in coverage:
                    if rec["circuit_id"] == cid:
                        rec["served_from_previous_build"] = True
                        rec["note"] = "acquisition failed this run; serving last known-good geometry"
                served_prev.append(cid)
            except Exception:
                pass
    keep = {c["circuit_id"] for c in circuits_index}
    # Copy staged circuit files over; remove only files that are neither freshly built nor carried forward.
    for f in (STAGE / "circuits").glob("*.json"):
        shutil.copy2(f, OUT / "circuits" / f.name)
    for f in (OUT / "circuits").glob("*.json"):
        if f.stem not in keep:
            f.unlink()
            print(f"[prune] removed obsolete {f.name} (no source this run and no carry-forward)", flush=True)
    return served_prev


def _validate_dataset(root, circuits_index, event_to_circuit) -> list[str]:
    """Validate a dataset dir (staging or promoted): index schema, every circuit file,
    identity mapping consistency, and that files + index refer to the same circuits."""
    problems = []
    try:
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        assert "coverage" in idx and "event_to_circuit" in idx["circuits"]
        # identity: Barcelona and Madrid must never resolve to the same circuit
        m = idx["circuits"]["event_to_circuit"]
        if m.get("Barcelona Grand Prix") and m.get("Spanish Grand Prix") and \
           m["Barcelona Grand Prix"] == m["Spanish Grand Prix"]:
            problems.append("identity error: Barcelona GP and Spanish GP map to the same circuit")
    except Exception as e:
        problems.append(f"index.json invalid: {e}")
    for c in circuits_index:
        f = root / "circuits" / f"{c['circuit_id']}.json"
        if not f.is_file():
            problems.append(f"{c['circuit_id']}: index lists it available but file missing"); continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"{c['circuit_id']}: unreadable ({e})"); continue
        for p in validate_geometry(d):
            problems.append(f"{c['circuit_id']}: {p}")
        if d.get("geometry_type") == "osm_centreline" and not (d.get("osm") or {}).get("way_ids"):
            problems.append(f"{c['circuit_id']}: OSM geometry missing way-id provenance")
    return problems


def _write_osm_attribution(circuits_index):
    osm = [c for c in circuits_index if c.get("geometry_type") == "osm_centreline"]
    lines = [
        "# OpenStreetMap attribution & licensing (ODbL 1.0)", "",
        "Some circuit outlines in this project are derived from OpenStreetMap data.", "",
        "- **Attribution:** © OpenStreetMap contributors.",
        "- **Licence:** Open Database License (ODbL) v1.0 — https://opendatacommons.org/licenses/odbl/1-0/",
        "- **Copyright page:** https://www.openstreetmap.org/copyright", "",
        "## Share-alike notice", "",
        "Attribution alone is **not** full compliance. OpenStreetMap data is licensed under the ODbL, whose",
        "share-alike and keep-open terms apply when a *Derivative Database* is publicly used or redistributed.",
        "The normalised circuit outlines under `degradiq/web/data/circuits/` that are marked",
        "`geometry_type: osm_centreline` are a Derivative Database of OpenStreetMap and are made available under",
        "the same ODbL 1.0 terms. Each such file records the exact OSM way ids, source URLs, the Overpass query",
        "and the retrieval timestamp used to produce it.", "",
        "FastF1-derived outlines are a separate source and are **not** OSM data.", "",
        "## Circuits using OpenStreetMap geometry", "",
    ]
    for c in osm:
        lines.append(f"- **{c['name']}** (`{c['circuit_id']}`) — see the file's `osm.way_ids` / `osm.source_urls`.")
    if not osm:
        lines.append("- (none in the current build)")
    (OUT / "OSM_ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _keep_existing_tyre():
    """When running --circuits only, preserve the tyre section already in index.json."""
    p = OUT / "index.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("tyre", {"season": TYRE_SEASON, "events": []})
        except Exception:
            pass
    return {"season": TYRE_SEASON, "events": []}


def _write_coverage_report(index):
    cov = index["coverage"]
    lines = ["# Circuit Intelligence — geometry coverage report", "",
             f"_Generated {index['generated']}_", "",
             "Four classifications — a PARTIAL map is never treated as complete:", "",
             "- **VERIFIED** — FastF1 racing-line trace with real corner numbers & marshal sectors.",
             "- **PARTIAL** — validated OpenStreetMap closed-loop centreline outline; no corners/sectors.",
             "- **UNAVAILABLE** — no cached telemetry and no usable OSM circuit loop; honest empty state.",
             "- **FAILED** — a source was attempted but acquisition/validation failed.", "",
             f"- Supported events: **{cov['supported_events']}**",
             f"- Circuit configurations: **{cov['circuits_total']}**",
             f"- VERIFIED: **{cov['verified']}**  ·  PARTIAL: **{cov['partial']}**  ·  "
             f"UNAVAILABLE: **{cov['unavailable']}**  ·  FAILED: **{cov['failed']}**", "",
             "## Per-circuit status", "", "| Circuit | Status | Geometry | Detail |", "|---|---|---|---|"]
    for c in cov["circuits"]:
        detail = c.get("reason") or (f"{c.get('track_length_m')} m loop" if c.get("track_length_m") else "")
        lines.append(f"| {c['circuit_id']} | {c['status']} | {c.get('geometry_type') or '—'} | {detail} |")
    lines += ["", "## Per-event coverage (2026 calendar)", "", "| Round | Event | Circuit | Status | Geometry |",
              "|---|---|---|---|---|"]
    for e in cov["events"]:
        lines.append(f"| {e['round']} | {e['event']} | {e['circuit_id'] or '—'} | {e['status']} | {e['geometry_type'] or '—'} |")
    lines += ["", "## Geometry sources & licensing", "",
              "- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised "
              "outline and corner numbers are packaged; raw telemetry is not redistributed.",
              f"- **OpenStreetMap** centrelines: {OSM_LICENCE}. The derived outlines are an ODbL Derivative Database "
              "(share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, "
              "the Overpass query and retrieval time.", ""]
    (OUT / "coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(circuits_only="--circuits" in sys.argv)
