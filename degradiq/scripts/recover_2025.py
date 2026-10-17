"""Recover remaining historical 2025 sessions using RateLimiter and CheckpointManager."""
import logging
from pathlib import Path
import sys
import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.rate_limiter import RateLimiter, install_rate_limiter, CheckpointManager
from degradiq.pit_strategy import build_hazard_rows, PACKAGE_DIR

CACHE = PACKAGE_DIR / "cache" / "fastf1"
CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

limiter = RateLimiter(min_interval_s=9.0)
install_rate_limiter(limiter)
mgr = CheckpointManager(PACKAGE_DIR / "cache" / "hazard_checkpoints")

schedule = fastf1.get_event_schedule(2025, include_testing=False)

for _, event in schedule.iterrows():
    name = str(event.get("EventName", "Unknown"))
    round_id = event.get("RoundNumber", name)
    if pd.isna(round_id) or round_id == 0:
        continue
    round_id = int(round_id)
    
    # Check if checkpoint already exists
    if mgr.checkpoint_path(2025, round_id).exists():
        print(f"[Skip] 2025 Round {round_id} ({name}) already checkpointed.")
        continue
        
    print(f"\n[Recovering] 2025 Round {round_id} ({name})...")
    try:
        session = fastf1.get_session(2025, round_id, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        results = getattr(session, "results", pd.DataFrame())
        max_lap = pd.to_numeric(laps.get("LapNumber", pd.Series(dtype=float)), errors="coerce").max()
        race_distance = int(max_lap) if pd.notna(max_lap) else None
        rows = build_hazard_rows(laps, results, name, race_distance)
        n_stops = int(rows["StopNextLap"].sum()) if not rows.empty else 0
        mgr.save_checkpoint(2025, round_id, name, rows, status="loaded")
        mgr.save_checkpoint(2025, name.replace(" ", "_"), name, rows, status="loaded")
        print(f"[Success] 2025 Round {round_id} ({name}): {len(rows)} rows, {n_stops} stops.")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"[Failed] 2025 Round {round_id} ({name}): {err}")
        mgr.save_checkpoint(2025, round_id, name, pd.DataFrame(), status="unavailable", error=err)
        mgr.save_checkpoint(2025, name.replace(" ", "_"), name, pd.DataFrame(), status="unavailable", error=err)

print("\n2025 recovery sweep complete.")
