"""Configuration for the DegradIQ 2026 season-wide rebuild."""
from __future__ import annotations

YEAR = 2026
SESSIONS = ("FP1", "FP2", "FP3")
RACE_SESSION = "R"

# Full-season 2026 driver lineup, using FastF1 three-letter driver codes.
# Substitute appearances are intentionally excluded.
FULL_SEASON_DRIVERS = frozenset({
    "RUS", "ANT", "LEC", "HAM", "NOR", "PIA", "VER", "HAD",
    "LAW", "LIN", "GAS", "COL", "OCO", "BEA", "HUL", "BOR",
    "SAI", "ALB", "ALO", "STR", "PER", "BOT",
})

# Physical assumptions retained from the project brief.
FUEL_BURN_KG_PER_LAP = 1.9
SECONDS_PER_KG = 0.03
FUEL_CORRECTION_SECONDS_PER_LAP = FUEL_BURN_KG_PER_LAP * SECONDS_PER_KG
MIN_TYRE_LIFE = 4
TRAFFIC_OUTLIER_SECONDS = 1.5
TRACK_EVOLUTION_MAX_STINT_RANGE_FRACTION = 1.0
RANDOM_STATE = 42

PIT_THRESHOLDS_SECONDS = (1.5, 2.5)
MIN_MODEL_ROWS = 8
MIN_UNIQUE_TYRE_AGES = 4
CACHE_DIR = "data"
OUTPUT_DIR = "degradiq/figures"

# 2026 full-season team mapping used for strategy features and API defaults.
DRIVER_TEAM_2026 = {
    "RUS": "Mercedes", "ANT": "Mercedes",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "VER": "Red Bull Racing", "HAD": "Red Bull Racing",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "GAS": "Alpine", "COL": "Alpine",
    "OCO": "Haas", "BEA": "Haas",
    "HUL": "Audi", "BOR": "Audi",
    "SAI": "Williams", "ALB": "Williams",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "PER": "Cadillac", "BOT": "Cadillac",
}
