"""FastAPI service for the 2026 DegradIQ calendar and strategy forecasts."""
from __future__ import annotations
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # Package import for local development.
    from .config import YEAR, FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
    from .pit_strategy import load_first_stop_model, predict_first_stops, METADATA_PATH
except ImportError:  # Vercel root directory set to ./degradiq.
    from config import YEAR, FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
    from pit_strategy import load_first_stop_model, predict_first_stops, METADATA_PATH

app = FastAPI(
    title="DegradIQ API",
    description="2026 F1 tyre degradation and first-pit-stop prediction service.",
    version="2026.2",
)


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


@app.get("/")
def root():
    return {
        "service": "DegradIQ API",
        "season": YEAR,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "calendar": "/calendar",
        "first_stop_status": "/strategy/first-stop/status",
        "first_stop_lineup": "/strategy/first-stop/lineup",
        "first_stop_predict": "POST /strategy/first-stop/predict",
    }


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
    bundle = load_first_stop_model()
    if bundle is None:
        return {
            "status": "model_not_trained",
            "ready": False,
            "message": "Run python -m degradiq.scripts.train_first_stop locally, then include degradiq/artifacts/first_stop_model.joblib in the deployment.",
        }
    return {
        "status": "ready",
        "ready": True,
        "trained_at_utc": bundle.get("trained_at_utc"),
        "training_rows": bundle.get("training_rows"),
        "positive_rows": bundle.get("positive_rows"),
        "positive_rate": bundle.get("positive_rate"),
        "metadata_available": METADATA_PATH.exists(),
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
    bundle = load_first_stop_model()
    if bundle is None:
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
    try:
        return predict_first_stops(bundle, request.event_name, request.race_laps, drivers, request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")
