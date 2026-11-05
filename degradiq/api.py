"""FastAPI service for the 2026 DegradIQ calendar and strategy forecasts."""
from __future__ import annotations
from typing import Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:  # Package import for local development.
    from .config import YEAR, FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
    from .prediction_engine import load_default_engine, PredictionEngine
except ImportError:  # Vercel root directory set to ./degradiq.
    from config import YEAR, FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
    from prediction_engine import load_default_engine, PredictionEngine

# The prediction engine loads the model artifact once and holds it in memory.
# Loading is lazy and fail-soft: a missing/incompatible artifact yields None and the
# endpoints return a clear 503 instead of fake predictions.
_ENGINE: "PredictionEngine | None" = None
_ENGINE_LOADED = False


def _engine() -> "PredictionEngine | None":
    global _ENGINE, _ENGINE_LOADED
    if not _ENGINE_LOADED:
        _ENGINE = load_default_engine()
        _ENGINE_LOADED = True
    return _ENGINE

app = FastAPI(
    title="DegradIQ API",
    description="2026 F1 tyre degradation and first-pit-stop prediction service.",
    version="2026.2",
)

# Browser frontend (PIT WALL) served from the same FastAPI origin. Assets are mounted
# only when present so the API still runs headless if the web bundle is absent.
_WEB_DIR = Path(__file__).resolve().parent / "web"
if (_WEB_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(_WEB_DIR / "assets")), name="assets")


class DriverForecastInput(BaseModel):
    driver: str = Field(..., min_length=3, max_length=3, description="2026 full-season driver code")
    grid_position: int = Field(..., ge=1, le=22)
    starting_compound: str = Field(default="MEDIUM", description="SOFT, MEDIUM, HARD, INTERMEDIATE or WET")
    starting_tyre_age: int = Field(default=1, ge=1, le=100)
    team: str | None = None
    current_lap: int | None = Field(default=None, ge=1)
    current_tyre_age: int | None = Field(default=None, ge=1, le=100)
    current_compound: str | None = None
    current_position: int | None = Field(default=None, ge=1, le=22)
    recent_pace_loss_s: float | None = Field(default=None, ge=-30, le=30)


class FirstStopRequest(BaseModel):
    event_name: str = Field(..., min_length=2, max_length=100)
    race_laps: int = Field(..., ge=5, le=100)
    mode: Literal["pre-race", "live"] = "pre-race"
    drivers: list[DriverForecastInput] = Field(..., min_length=1, max_length=22)


def _service_meta():
    return {
        "service": "DegradIQ API",
        "season": YEAR,
        "status": "ok",
        "app": "/",
        "docs": "/docs",
        "meta": "/api/meta",
        "health": "/health",
        "calendar": "/calendar",
        "first_stop_status": "/strategy/first-stop/status",
        "first_stop_lineup": "/strategy/first-stop/lineup",
        "first_stop_predict": "POST /strategy/first-stop/predict",
        "intel_index": "/api/intel/index",
        "intel_circuit": "/api/intel/circuit/{circuit_id}",
        "intel_tyre": "/api/intel/tyre/{event_id}",
    }


@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def web_app():
    """Serve the PIT WALL browser interface; fall back to service metadata JSON."""
    index = _WEB_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return JSONResponse(_service_meta())


@app.get("/api/meta")
def service_meta():
    return _service_meta()


@app.get("/health")
def health():
    return {"status": "ok", "season": YEAR}


import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CALENDAR_PATH = BASE_DIR / "calendar_2026.json"


@app.get("/calendar")
def calendar():
    if not CALENDAR_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Calendar snapshot missing at {CALENDAR_PATH.name}",
        )
    try:
        with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load calendar: {type(exc).__name__}: {exc}")


@app.get("/strategy/first-stop/status")
def first_stop_status():
    engine = _engine()
    if engine is None:
        return {
            "status": "model_not_trained",
            "ready": False,
            "message": "Run python -m degradiq.scripts.train_first_stop locally, then include degradiq/artifacts/first_stop_model.joblib in the deployment.",
        }
    return {
        "status": "ready",
        "ready": True,
        "provenance": engine.provenance(),
    }


@app.get("/strategy/first-stop/lineup")
def first_stop_lineup():
    return {
        "season": YEAR,
        "drivers": [
            {"driver": code, "team": DRIVER_TEAM_2026.get(code, "Unknown")}
            for code in sorted(FULL_SEASON_DRIVERS)
        ],
    }


@app.post("/strategy/first-stop/predict")
def first_stop_predict(request: FirstStopRequest):
    engine = _engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="First-stop model artifact is not available. Train it locally with python -m degradiq.scripts.train_first_stop and deploy the generated artifact.",
        )
    drivers = []
    seen = set()
    for item in request.drivers:
        code = item.driver.strip().upper()
        if code not in FULL_SEASON_DRIVERS:
            raise HTTPException(status_code=422, detail=f"{code} is not in the configured 2026 full-season lineup.")
        if code in seen:
            raise HTTPException(status_code=422, detail=f"Duplicate driver code: {code}.")
        seen.add(code)
        data = item.model_dump()
        data["driver"] = code
        data["starting_compound"] = data["starting_compound"].upper()
        if data.get("current_compound"):
            data["current_compound"] = data["current_compound"].upper()
        drivers.append(data)
    if request.mode == "live":
        missing = [d["driver"] for d in drivers if d.get("current_lap") is None]
        if missing:
            raise HTTPException(status_code=422, detail=f"Live mode requires current_lap for every driver. Missing: {', '.join(missing)}")
    context_mode = "live" if request.mode == "live" else "historical"
    try:
        return engine.predict(request.event_name, request.race_laps, drivers,
                              mode=request.mode, context_mode=context_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Circuit & Tyre Intelligence — read-only, packaged offline data only.
# Serves version-controlled JSON built by degradiq/scripts/build_intel_data.py from
# the local FastF1 cache. No live fetch, no filesystem traversal, no secret exposure.
# Circuits/events without cached data are reported as unavailable (never fabricated).
# ---------------------------------------------------------------------------
import re as _re

_DATA_DIR = _WEB_DIR / "data"
_ID_RE = _re.compile(r"^[a-z0-9-]{1,40}$")


def _load_intel_index() -> dict:
    path = _DATA_DIR / "index.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Intelligence data bundle is not deployed. Build it locally with python -m degradiq.scripts.build_intel_data.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load intel index: {type(exc).__name__}: {exc}")


def _safe_data_file(kind: str, item_id: str) -> Path:
    """Resolve a packaged data file by validated id; reject anything path-like."""
    if not _ID_RE.match(item_id):
        raise HTTPException(status_code=422, detail="Invalid id. Allowed: lowercase letters, digits and hyphens.")
    path = (_DATA_DIR / kind / f"{item_id}.json").resolve()
    root = (_DATA_DIR / kind).resolve()
    if root not in path.parents:  # defence-in-depth against traversal
        raise HTTPException(status_code=422, detail="Invalid id.")
    return path


@app.get("/api/intel/index")
def intel_index():
    """Availability map: which circuits have verified geometry and which events have tyre data."""
    idx = _load_intel_index()
    return {
        "status": "ok",
        "source": idx.get("source"),
        "generated": idx.get("generated"),
        "circuits": idx.get("circuits", {}),
        "tyre": idx.get("tyre", {}),
        "note": "Geometry and tyre observations are derived from the local FastF1 cache only; "
                "circuits/events without cached data are reported as unavailable, never substituted.",
    }


@app.get("/api/intel/circuit/{circuit_id}")
def intel_circuit(circuit_id: str):
    """Verified Grand Prix circuit geometry, or an honest unavailable state."""
    circuit_id = circuit_id.strip().lower()
    path = _safe_data_file("circuits", circuit_id)
    if not path.is_file():
        idx = _load_intel_index()
        return JSONResponse(status_code=404, content={
            "status": "unavailable", "circuit_id": circuit_id, "available": False,
            "reason": "No verified geometry for this circuit. Geometry requires cached FastF1 position telemetry, "
                      "which exists only for a limited set of circuits in this deployment.",
            "available_circuits": idx.get("circuits", {}).get("available", []),
        })
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load circuit: {type(exc).__name__}: {exc}")
    data["status"] = "ok"
    return data


@app.get("/api/intel/tyre/{event_id}")
def intel_tyre(event_id: str):
    """Real historical lap-time vs tyre-age observations for a cached event, or unavailable."""
    event_id = event_id.strip().lower()
    path = _safe_data_file("tyre", event_id)
    if not path.is_file():
        idx = _load_intel_index()
        return JSONResponse(status_code=404, content={
            "status": "unavailable", "event_id": event_id, "available": False,
            "reason": "No cached tyre data for this event in this deployment.",
            "available_events": idx.get("tyre", {}).get("events", []),
        })
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load tyre data: {type(exc).__name__}: {exc}")
    data["status"] = "ok"
    return data
