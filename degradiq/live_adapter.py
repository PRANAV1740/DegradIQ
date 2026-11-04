"""Data-source adapter boundary for first-stop prediction (Phase VI).

Separates *where the per-lap feature rows come from* from *how they are scored*. The
``PredictionEngine`` scores rows; a ``LiveSource`` supplies them and reports its own
availability and freshness. This lets the apps distinguish three regimes honestly:

  * **live** — a real-time feed (not provided here; see below).
  * **historical_replay** — point-in-time stepping through a recorded event.
  * **unavailable** — no feed; the UI must say so, never invent timing.

On live data: FastF1 provides *historical* timing, not a reliable real-time stream,
and the official F1 live-timing endpoint requires per-session authentication this
project does not have (an unauthenticated probe returns HTTP 401). So the default
live source is :class:`UnavailableLiveSource`, and the working real-time-like
experience is point-in-time replay via :class:`ReplayLiveSource`. No source in this
module fabricates timing, tyre data, or race events.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import pandas as pd

try:
    from .pit_strategy import FEATURE_COLUMNS
    from .prediction_engine import PredictionEngine
except ImportError:  # Vercel root set to the package dir.
    from pit_strategy import FEATURE_COLUMNS  # type: ignore
    from prediction_engine import PredictionEngine  # type: ignore

LIVE_UNAVAILABLE_REASON = (
    "No real-time feed: FastF1 exposes historical timing, not a reliable live stream, "
    "and the official live-timing endpoint requires per-session authentication "
    "(unauthenticated access returns HTTP 401). Use historical replay instead."
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveDataStatus:
    """What a source can currently provide, for honest UI/freshness indicators."""
    available: bool
    source: str                 # "live" | "historical_replay" | "unavailable"
    reason: str
    as_of_utc: str
    current_lap: int | None = None
    total_laps: int | None = None


@runtime_checkable
class LiveSource(Protocol):
    def status(self) -> LiveDataStatus: ...
    def feature_rows(self) -> pd.DataFrame | None:
        """Per-lap feature rows available *now* (never future laps), or None."""
        ...


class UnavailableLiveSource:
    """The honest default: no live feed; returns no rows and says why."""

    def __init__(self, reason: str = LIVE_UNAVAILABLE_REASON):
        self.reason = reason

    def status(self) -> LiveDataStatus:
        return LiveDataStatus(False, "unavailable", self.reason, _utcnow())

    def feature_rows(self) -> pd.DataFrame | None:
        return None


class ReplayLiveSource:
    """Point-in-time replay over a recorded per-lap feature frame.

    The cursor is the "current" lap; ``feature_rows`` returns only rows at or before
    the cursor, so a consumer can never see future laps — the same guarantee the
    dashboard replay relies on. The frame must carry the current ``FEATURE_COLUMNS``
    (plus optionally ``StopNextLap`` for comparison).
    """

    def __init__(self, frame: pd.DataFrame, event_name: str | None = None):
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"ReplayLiveSource frame is missing feature columns: {missing}")
        self._frame = frame.sort_values("LapNumber").reset_index(drop=True)
        self.event_name = event_name or (
            str(self._frame["EventName"].iloc[0]) if "EventName" in self._frame.columns
            and len(self._frame) else "Unknown")
        laps = self._frame["LapNumber"].astype(int)
        self._min_lap = int(laps.min()) if len(laps) else 0
        self._max_lap = int(laps.max()) if len(laps) else 0
        self.cursor = self._min_lap

    def seek(self, lap: int) -> "ReplayLiveSource":
        self.cursor = max(self._min_lap, min(int(lap), self._max_lap))
        return self

    def advance(self, laps: int = 1) -> "ReplayLiveSource":
        return self.seek(self.cursor + laps)

    @property
    def finished(self) -> bool:
        return self.cursor >= self._max_lap

    def status(self) -> LiveDataStatus:
        return LiveDataStatus(True, "historical_replay",
                              f"Replaying {self.event_name} up to lap {self.cursor}.",
                              _utcnow(), current_lap=self.cursor, total_laps=self._max_lap)

    def feature_rows(self) -> pd.DataFrame | None:
        rows = self._frame[self._frame["LapNumber"].astype(int) <= self.cursor]
        return rows if len(rows) else None

    def actual_first_stop_lap(self, driver: str) -> int | None:
        if "StopNextLap" not in self._frame.columns:
            return None
        d = self._frame[(self._frame.get("Driver") == driver)
                        & (self._frame["StopNextLap"] == 1)]
        return int(d["LapNumber"].iloc[0]) + 1 if len(d) else None


def refresh_prediction(engine: PredictionEngine, source: LiveSource) -> dict:
    """Score whatever the source can currently provide, with its status attached.

    Returns the source status and, when rows are available, the same rows with a
    per-lap ``hazard`` column. Never fabricates rows when the source has none.
    """
    status = source.status()
    rows = source.feature_rows()
    if rows is None or not len(rows):
        return {"status": status, "scored": None}
    scored = rows.copy()
    scored["hazard"] = engine.score_rows(rows)
    return {"status": status, "scored": scored}
