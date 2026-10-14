"""Stage 1: discover the 2026 calendar and ingest FastF1 lap data."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import fastf1

try:
    from .config import YEAR, SESSIONS, RACE_SESSION, FULL_SEASON_DRIVERS, MIN_TYRE_LIFE
except ImportError:  # Vercel root directory may be this package directory.
    from config import YEAR, SESSIONS, RACE_SESSION, FULL_SEASON_DRIVERS, MIN_TYRE_LIFE


def _seconds(series: pd.Series) -> pd.Series:
    """Convert timedelta-like FastF1 values to seconds without raising on NaT."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def get_calendar(year: int = YEAR) -> pd.DataFrame:
    """Return the complete calendar FastF1 currently knows, including future events."""
    schedule = fastf1.get_event_schedule(year, include_testing=False).copy()
    # FastF1 versions vary slightly in schedule column names. Preserve source fields.
    if "EventName" not in schedule.columns:
        raise ValueError("FastF1 event schedule has no EventName column.")
    if "EventDate" in schedule.columns:
        schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], errors="coerce", utc=True).dt.tz_convert(None)
    elif "Session5Date" in schedule.columns:
        schedule["EventDate"] = pd.to_datetime(schedule["Session5Date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        schedule["EventDate"] = pd.NaT
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    schedule["CalendarStatus"] = np.where(
        schedule["EventDate"].notna() & (schedule["EventDate"] > now),
        "Pending",
        "Date passed, check session availability",
    )
    schedule["Season"] = year
    return schedule


def full_season_driver_filter(laps: pd.DataFrame) -> pd.DataFrame:
    """Restrict data to the confirmed full-season driver lineup."""
    if "Driver" not in laps.columns:
        return laps.iloc[0:0].copy()
    return laps[laps["Driver"].isin(FULL_SEASON_DRIVERS)].copy()


def normalize_laps(laps: pd.DataFrame, session_name: str, event_name: str,
                   year: int, session_date=None) -> pd.DataFrame:
    """Standardize FastF1 laps and apply common validity filters."""
    df = laps.copy()
    for src, dst in (("LapTime", "LapTime_s"), ("Sector1Time", "Sector1Time_s"),
                     ("Sector2Time", "Sector2Time_s"), ("Sector3Time", "Sector3Time_s")):
        if src in df.columns:
            df[dst] = _seconds(df[src])
        else:
            df[dst] = np.nan

    df["Session"] = session_name
    df["EventName"] = event_name
    df["Season"] = year
    if session_date is not None:
        df["SessionDate"] = pd.to_datetime(session_date, errors="coerce")
    if "SessionTime" in df.columns:
        df["SessionMinutes"] = _seconds(df["SessionTime"]) / 60.0
    elif "Time" in df.columns:
        df["SessionMinutes"] = _seconds(df["Time"]) / 60.0
    else:
        df["SessionMinutes"] = np.nan

    required = ["LapTime_s", "Driver", "Compound", "TyreLife", "Stint", "LapNumber"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"].fillna(False).astype(bool)]
    if "PitInTime" in df.columns and "PitOutTime" in df.columns:
        df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"].astype(str) == "1"]
    df = full_season_driver_filter(df)
    df = df[pd.to_numeric(df["TyreLife"], errors="coerce") >= MIN_TYRE_LIFE]
    df = df[pd.to_numeric(df["LapTime_s"], errors="coerce") > 0]
    return df.copy()


import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_fastf1_cache_dir(configured_dir: str | Path = "data") -> Path:
    """Resolve the FastF1 cache path.

    - Under Vercel serverless (os.environ.get('VERCEL')), explicitly routes to /tmp/fastf1_cache.
    - In local development, resolves relative paths against the project root (preserving P:\\DegradIQ\\data).
    """
    if os.environ.get("VERCEL"):
        cache_path = Path("/tmp/fastf1_cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    path_obj = Path(configured_dir)
    if not path_obj.is_absolute():
        cache_path = (PROJECT_ROOT / path_obj).resolve()
    else:
        cache_path = path_obj.resolve()

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_event_session(year: int, event_identifier, session_name: str,
                       cache_dir: str | Path = "data") -> pd.DataFrame:
    """Load a single event/session. Errors are handled by the season orchestrator."""
    cache_path = resolve_fastf1_cache_dir(cache_dir)
    fastf1.Cache.enable_cache(str(cache_path))
    session = fastf1.get_session(year, event_identifier, session_name)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return normalize_laps(
        session.laps, session_name, str(getattr(session.event, "EventName", event_identifier)),
        year, getattr(session, "date", None)
    )


def load_event_practice(year: int, event_identifier, cache_dir: str = "data") -> tuple[pd.DataFrame, list[dict]]:
    frames, statuses = [], []
    for name in SESSIONS:
        try:
            frame = load_event_session(year, event_identifier, name, cache_dir)
            frames.append(frame)
            statuses.append({"session": name, "status": "loaded", "laps": len(frame), "error": ""})
        except Exception as exc:
            statuses.append({"session": name, "status": "unavailable", "laps": 0,
                             "error": f"{type(exc).__name__}: {exc}"})
    if not frames:
        return pd.DataFrame(), statuses
    return pd.concat(frames, ignore_index=True), statuses


def load_race(year: int, event_identifier, cache_dir: str = "data") -> tuple[pd.DataFrame, str]:
    try:
        frame = load_event_session(year, event_identifier, RACE_SESSION, cache_dir)
        return frame, "loaded"
    except Exception as exc:
        return pd.DataFrame(), f"unavailable: {type(exc).__name__}: {exc}"
