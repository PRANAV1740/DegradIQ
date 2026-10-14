"""Batch runner for every completed 2026 event in the current FastF1 calendar.

Usage from repository root:
    python -m degradiq.scripts.run_season

All generated reports stay inside degradiq/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.config import YEAR, CACHE_DIR
from degradiq.ingest import get_calendar, load_event_practice, load_race
from degradiq.validation import validate_event

OUT = Path(__file__).resolve().parents[1]
OUT.mkdir(parents=True, exist_ok=True)


def main():
    calendar = get_calendar(YEAR)
    coverage, validations = [], []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    for _, event in calendar.iterrows():
        event_name = str(event["EventName"])
        event_date = event.get("EventDate", pd.NaT)
        event_id = event.get("RoundNumber", event_name)
        if pd.isna(event_id) or event_id == 0:
            event_id = event_name
        if pd.notna(event_date) and pd.Timestamp(event_date).tz_localize(None) > now:
            coverage.append({"season": YEAR, "event": event_name, "status": "Pending",
                             "practice_laps": 0, "race_status": "Pending"})
            continue

        print(f"\n=== {YEAR} {event_name} ===")
        practice, practice_statuses = load_event_practice(YEAR, event_id, CACHE_DIR)
        race, race_status = load_race(YEAR, event_id, CACHE_DIR)
        coverage.append({"season": YEAR, "event": event_name,
                         "status": "data available" if not practice.empty else "no eligible practice data",
                         "practice_laps": len(practice), "race_status": race_status,
                         "practice_sessions": str(practice_statuses)})
        if practice.empty or race.empty:
            continue
        rows, _ = validate_event(practice, race)
        for _, row in rows.iterrows():
            validations.append({"season": YEAR, "event": event_name, **row.to_dict()})

    pd.DataFrame(coverage).to_csv(OUT / f"season_coverage_{YEAR}.csv", index=False)
    pd.DataFrame(validations).to_csv(OUT / f"validation_{YEAR}.csv", index=False)
    print(f"\nCoverage report: {OUT / f'season_coverage_{YEAR}.csv'}")
    print(f"Validation report: {OUT / f'validation_{YEAR}.csv'}")


if __name__ == "__main__":
    main()
