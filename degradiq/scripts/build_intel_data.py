"""Offline builder for Circuit Intelligence + Tyre Intelligence data.

Produces packaged, version-controlled JSON under degradiq/web/data/ from the LOCAL
FastF1 cache only — no fabricated geometry or tyre values. Circuit geometry is derived
from a session's position telemetry + FastF1's circuit_info (corners, marshal sectors,
rotation); tyre observations are real lap times vs tyre age after the project's standard
cleaning. Circuits/events without cached data are simply omitted and the app shows an
honest unavailable state.

Run locally (needs the FastF1 cache), then commit degradiq/web/data/:

    python -m degradiq.scripts.build_intel_data
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fastf1  # noqa: E402
from degradiq.pit_strategy import PACKAGE_DIR  # noqa: E402

fastf1.Cache.enable_cache(str(PACKAGE_DIR / "cache" / "fastf1"))
OUT = PACKAGE_DIR / "web" / "data"
NOW = datetime.now(timezone.utc).date().isoformat()

# Circuits with cached position telemetry -> verified geometry.
GEOMETRY_EVENTS = [(2024, "Bahrain Grand Prix"), (2024, "Spanish Grand Prix"), (2024, "Italian Grand Prix")]
# Events with cached laps -> real tyre observations (2024 season for broad, bounded coverage).
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


def _slug(name: str) -> str:
    return name.lower().replace(" grand prix", "").replace(" ", "-").replace("ã", "a").replace("ç", "c")


def build_circuit(year: int, name: str) -> dict | None:
    s = fastf1.get_session(year, name, "R")
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
    allx = np.concatenate([X, cx]); ally = np.concatenate([Y, cy])
    minx, maxx, miny, maxy = allx.min(), allx.max(), ally.min(), ally.max()
    span = max(maxx - minx, maxy - miny) or 1.0

    def norm(x, y):  # fit into 0..1000 with a small margin, Y flipped for SVG
        return [round((x - minx) / span * 940 + 30, 1), round((maxy - y) / span * 940 + 30, 1)]

    # Downsample the outline to keep the SVG light.
    step = max(1, len(X) // 360)
    outline = [norm(X[i], Y[i]) for i in range(0, len(X), step)]
    corners = [{"number": int(r["Number"]), "xy": norm(cx[i], cy[i])} for i, (_, r) in enumerate(ci.corners.iterrows())]
    # Marshal sectors (FastF1): distance thresholds -> which telemetry point starts each sector.
    sectors = []
    try:
        ms = ci.marshal_sectors
        if ms is not None and len(ms):
            sectors = [{"number": int(r["Number"]), "xy": norm(*rotate(float(r["X"]), float(r["Y"])))} for _, r in ms.iterrows()]
    except Exception:
        sectors = []
    return {
        "circuit_id": _slug(name), "name": name,
        "location": str(getattr(s.event, "Location", "")) or None,
        "country": str(getattr(s.event, "Country", "")) or None,
        "source": "FastF1 session telemetry (fastest-lap track path + circuit_info corners/marshal sectors/rotation)",
        "source_event": f"{year} {name} · Race", "retrieved": NOW,
        "licence": "Derived from FastF1-cached F1 timing/telemetry for research use; not for redistribution of raw F1 data.",
        "coordinate_system": "normalised 0-1000 viewBox, circuit rotation applied, Y flipped for SVG",
        "available": {"outline": True, "corners": len(corners) > 0, "marshal_sectors": len(sectors) > 0, "pit_lane": False},
        "outline": outline, "corners": corners, "marshal_sectors": sectors,
        "start_finish": outline[0] if outline else None,
        "n_outline_points": len(outline),
        "notes": "Outline is the fastest lap's GPS path (approximate racing line, not a surveyed centreline). "
                 "Pit-lane geometry is not derived. Corner numbers and marshal sectors are from FastF1 circuit_info.",
    }


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()
    keep = df["LapTime_s"].notna() & df["Compound"].notna() & df["TyreLife"].notna()
    keep &= df["PitInTime"].isna() & df["PitOutTime"].isna()          # exclude in/out laps
    if "IsAccurate" in df.columns:
        keep &= df["IsAccurate"].fillna(False)                         # FastF1 accuracy gate
    if "TrackStatus" in df.columns:
        ts = df["TrackStatus"].astype(str)
        keep &= ~ts.str.contains("4") & ~ts.str.contains("5") & ~ts.str.contains("6") & ~ts.str.contains("7")  # SC/red/VSC
    return df[keep]


def build_tyre(year: int, name: str) -> dict | None:
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
            # Robust to stragglers: drop lap times > 107% of the age-group median before summarising.
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
            r["pace_loss_s"] = round(r["median_s"] - base, 3)  # vs this compound's youngest clean age here
        compounds[comp] = {"color": COMPOUND_COLOR[comp], "n": int(len(g)),
                           "age_range": [rows[0]["age"], rows[-1]["age"]], "observations": rows}
    if not compounds:
        return None
    return {
        "event_id": _slug(name), "event": name, "season": year, "session": "Race",
        "source": "FastF1-cached race lap timing", "retrieved": NOW,
        "metric": "actual lap time (seconds), no fuel/traffic correction — raw observations",
        "cleaning": "excluded in/out laps, IsAccurate==False, and laps under Safety-Car/VSC/red-flag track status; "
                    "per tyre-age group dropped lap times > 107% of the group median.",
        "limitations": "Lap time is affected by fuel load, traffic, track evolution and driver — not tyre age alone. "
                       "Pace-loss is relative to the youngest clean age shown, not a validated degradation model. "
                       "No tyre temperature/pressure/wear is measured.",
        "total_laps": int(total), "clean_laps": int(len(clean)), "compounds": compounds,
    }


def main() -> None:
    (OUT / "circuits").mkdir(parents=True, exist_ok=True)
    (OUT / "tyre").mkdir(parents=True, exist_ok=True)
    circuits, tyre = [], []
    for year, name in GEOMETRY_EVENTS:
        try:
            c = build_circuit(year, name)
            if c:
                (OUT / "circuits" / f"{c['circuit_id']}.json").write_text(json.dumps(c), encoding="utf-8")
                circuits.append({"circuit_id": c["circuit_id"], "name": name, "available": True})
                print(f"[circuit] {name}: {c['n_outline_points']} pts, {len(c['corners'])} corners")
        except Exception as e:
            print(f"[circuit] {name}: FAILED {type(e).__name__}: {e}")
    for name in TYRE_EVENTS:
        try:
            t = build_tyre(TYRE_SEASON, name)
            if t:
                (OUT / "tyre" / f"{t['event_id']}.json").write_text(json.dumps(t), encoding="utf-8")
                tyre.append({"event_id": t["event_id"], "event": name, "season": TYRE_SEASON,
                             "compounds": list(t["compounds"].keys()), "clean_laps": t["clean_laps"]})
                print(f"[tyre] {name}: {t['clean_laps']} clean laps, {list(t['compounds'])}")
        except Exception as e:
            print(f"[tyre] {name}: FAILED {type(e).__name__}: {e}")
    index = {"generated": NOW, "source": "FastF1 cache (offline)",
             "circuits": {"available": circuits, "note": "Verified geometry exists only for circuits with cached position telemetry; others show an unavailable state."},
             "tyre": {"season": TYRE_SEASON, "events": tyre}}
    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {len(circuits)} circuits, {len(tyre)} tyre events -> {OUT}")


if __name__ == "__main__":
    main()
