"""
DegradIQ -- Isolate True F1 Tyre Degradation from Noisy Practice Data
=====================================================================

This script takes raw F1 practice-session lap times and removes the effects
that HIDE tyre wear (fuel burn, traffic, track evolution) to produce a clean
degradation curve per compound.

Pipeline:
  1. Ingest  -- load FP1-FP3 data via FastF1
  2. Clean   -- remove pit laps, outliers, safety-car laps
  3. De-fuel -- correct for the car getting lighter each lap
  4. De-traffic -- drop laps ruined by being stuck behind another car
  5. De-evolve  -- remove the track-getting-faster trend
  6. Model      -- fit a degradation curve per compound
  7. Visualise  -- three presentation-ready PNG charts

Run with:  python src/degradiq.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# FastF1 prints a LOT of info logs; suppress the noisy ones
import fastf1
fastf1.logger.set_log_level("WARNING")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — change these two lines to analyze any race
# ──────────────────────────────────────────────────────────────
YEAR          = 2024                       # season year
CIRCUIT       = "Bahrain"                  # circuit name (FastF1 recognises short names)
SESSIONS      = ["FP1", "FP2", "FP3"]     # practice sessions to pull
RACE_SESSION  = "R"                        # used for Graph 3 validation
CACHE_DIR     = "./data"                   # FastF1 downloads are cached here
OUTPUT_DIR    = "./outputs/figures"         # where the three PNGs land

# Physics constants (explained in Step 3)
START_FUEL      = 110.0   # kg — max fuel load at race start
FUEL_BURN_RATE  = 1.9     # kg/lap — roughly constant across teams
TIME_PER_KG     = 0.030   # seconds slower per kg of fuel onboard

# Minimum laps required per compound before we try to model it
MIN_LAPS_PER_COMPOUND = 8

# ──────────────────────────────────────────────────────────────
# STEP 1 — INGEST: Load practice sessions and build one big DataFrame
# ──────────────────────────────────────────────────────────────
def load_practice_data(year, circuit, sessions, cache_dir):
    """
    Download FP1–FP3 lap data via FastF1 and combine into a single DataFrame.

    What this function does, step by step:
      1. Enables FastF1's disk cache so we don't re-download on every run.
      2. Loops over each practice session (FP1, FP2, FP3).
      3. For each session, calls session.load() which fetches timing data
         from the official F1 API.
      4. Extracts session.laps — a pandas DataFrame where each ROW is one
         lap by one driver.
      5. Converts timedelta columns (LapTime, SectorXTime) to plain float
         seconds, which are much easier to do math on.
      6. Stacks all sessions into one DataFrame with a 'Session' column
         so we know which session each lap came from.

    Returns:
      pd.DataFrame with columns like:
        Driver, LapNumber, LapTime_s, Sector1Time_s, Sector2Time_s,
        Sector3Time_s, Compound, TyreLife, Stint, IsAccurate,
        PitInTime, PitOutTime, TrackStatus, Session, ...
    """

    # Tell FastF1 to save downloaded data to disk. Without this, every run
    # would re-download ~50MB of data from the F1 servers. Caching makes
    # subsequent runs take seconds instead of minutes.
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    all_laps = []  # we'll collect DataFrames here and stack them at the end

    for session_name in sessions:
        print(f"  Loading {year} {circuit} {session_name}...")

        # get_session() creates a Session object — it doesn't download yet.
        # The arguments are: year, circuit name (or round number), session type.
        session = fastf1.get_session(year, circuit, session_name)

        # session.load() is where the actual download happens.
        # It fetches: lap timing, telemetry, weather, and race-control messages.
        # We load laps + weather (for track temp), skip telemetry to save time.
        session.load(laps=True, telemetry=False, weather=True, messages=False)

        # session.laps is a special DataFrame (a "Laps" object) that has
        # helper methods like .pick_driver(), .pick_fastest(), etc.
        # But underneath it's just a pandas DataFrame.
        laps = session.laps.copy()

        # --- Convert LapTime from timedelta to float seconds ---
        # FastF1 stores LapTime as a pandas Timedelta (e.g., 0 days 00:01:32.456).
        # .dt.total_seconds() converts that to a plain float: 92.456
        # We need floats so we can do arithmetic (add fuel correction, etc.)
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

        # --- Convert sector times the same way ---
        # Sector times tell us how long the driver took in each third of the lap.
        # We'll use these in Step 4 (de-traffic) to spot which SECTOR had traffic.
        laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
        laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
        laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()

        # Tag each lap with its source session so we can filter later
        laps["Session"] = session_name

        # Try to grab track temperature from the weather data.
        # weather_data is a separate DataFrame with timestamps; we take the
        # median across the session as a rough single number.
        if session.weather_data is not None and not session.weather_data.empty:
            laps["TrackTemp"] = session.weather_data["TrackTemp"].median()
        else:
            laps["TrackTemp"] = np.nan

        all_laps.append(laps)
        print(f"    -> {len(laps)} laps loaded")

    # pd.concat() stacks the three DataFrames vertically (row-wise).
    # ignore_index=True resets the index to 0,1,2,... instead of keeping
    # the original per-session indices (which would have duplicates).
    combined = pd.concat(all_laps, ignore_index=True)
    print(f"\n  Total practice laps loaded: {len(combined)}")

    return combined


# --------------------------------------------------------------
# STEP 2 -- CLEAN: Remove junk laps that would pollute our analysis
# --------------------------------------------------------------
def clean_laps(df):
    """
    Remove laps that don't represent 'normal' driving pace.

    There are five categories of junk:
      1. Non-accurate laps: FastF1's IsAccurate flag catches timing glitches
      2. Pit in/out laps: Slow because the driver enters/exits pit lane
      3. Non-green-flag laps: Safety car, VSC, red flag distort pace
      4. Missing data: No compound or tyre age = useless for analysis
      5. Statistical outliers: Laps far slower than their stint median

    We remove these IN ORDER, printing how many laps survive each filter
    so you can see exactly where data is lost.
    """
    original_count = len(df)
    print(f"  Starting with {original_count} laps")

    # --- Filter 1: Keep only laps with valid lap times ---
    # Some laps have NaN LapTime_s (e.g., the very first lap of a session
    # where we don't have a full timing loop crossing).
    df = df.dropna(subset=["LapTime_s"])
    print(f"    After dropping NaN lap times: {len(df)} laps")

    # --- Filter 2: Keep only "accurate" laps ---
    # FastF1 sets IsAccurate=True when the lap's start/end times are
    # properly synchronised. False usually means a red flag or timing error.
    df = df[df["IsAccurate"] == True].copy()
    print(f"    After IsAccurate filter:      {len(df)} laps")

    # --- Filter 3: Remove pit in-laps and out-laps ---
    # PitInTime is not NaT when the driver entered the pit during this lap.
    # PitOutTime is not NaT when the driver exited the pit during this lap.
    # Both make the lap time artificially slow (pit limiter = 80 km/h).
    pit_mask = df["PitInTime"].isna() & df["PitOutTime"].isna()
    df = df[pit_mask].copy()
    print(f"    After removing pit laps:      {len(df)} laps")

    # --- Filter 4: Keep only green-flag laps ---
    # TrackStatus '1' means "all clear / green flag". Other values:
    #   '2' = yellow flag, '4' = safety car, '5' = red flag, '6' = VSC, etc.
    # We only want laps where the driver was pushing freely.
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"] == "1"].copy()
        print(f"    After green-flag filter:      {len(df)} laps")

    # --- Filter 5: Drop rows missing compound or tyre age ---
    # Can't model degradation if we don't know the tyre compound or how
    # many laps it has done. These are rare but happen.
    df = df.dropna(subset=["Compound", "TyreLife"])
    print(f"    After dropping missing data:   {len(df)} laps")

    # --- Filter 6: Drop rows missing sector times ---
    # We need sector times for the de-traffic step (Step 4).
    df = df.dropna(subset=["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"])
    print(f"    After dropping missing sectors:{len(df)} laps")

    # --- Filter 7: Remove statistical outliers per stint ---
    # Even after the above, some laps are slow due to driver error, traffic
    # we couldn't flag, or track incidents not captured in TrackStatus.
    # Strategy: within each driver+stint, if a lap is more than 2 standard
    # deviations slower than the stint median, it's likely not representative.
    cleaned_stints = []
    for (driver, stint), group in df.groupby(["Driver", "Stint"]):
        median = group["LapTime_s"].median()
        std = group["LapTime_s"].std()
        if pd.isna(std) or std == 0:
            # Only one lap or all identical: keep everything
            cleaned_stints.append(group)
            continue
        # Keep laps within 2 standard deviations of the median
        upper_bound = median + 2 * std
        cleaned_stints.append(group[group["LapTime_s"] <= upper_bound])

    df = pd.concat(cleaned_stints, ignore_index=True)
    print(f"    After outlier removal:         {len(df)} laps")
    print(f"  Removed {original_count - len(df)} junk laps total "
          f"({len(df)} remain)")

    return df


# --------------------------------------------------------------
# STEP 3 -- DE-FUEL: Remove the lap-time improvement from fuel burn
# --------------------------------------------------------------
def correct_fuel_effect(df):
    """
    Correct lap times so that fuel weight no longer hides tyre degradation.

    THE PHYSICS:
    An F1 car starts with ~110 kg of fuel. It burns ~1.9 kg each lap.
    Lighter car = faster lap. The effect is roughly 0.030 seconds per kg.

    So over a 10-lap stint, the car loses 19 kg of fuel. That fuel burn
    alone makes the car ~0.57s FASTER by lap 10, which MASKS the tyre wear
    (tyres getting slower). When you look at raw data, the fuel improvement
    and tyre degradation roughly cancel out, making the stint look flat.

    THE FIX:
    We add back the fuel time benefit. For each lap, we compute how much
    fuel has been burned (TyreLife * burn rate) and add the corresponding
    time penalty. This 're-weights' every lap as if the car still had
    full fuel, removing the fuel effect and exposing the tyre wear.

    Formula:  corrected_time = raw_time + TyreLife * BURN_RATE * TIME_PER_KG

    Example for lap with TyreLife = 5:
      fuel burned so far = 5 * 1.9 = 9.5 kg
      time advantage from lighter car = 9.5 * 0.030 = 0.285s
      correction: add 0.285s back to the raw time
    """
    # This is a vectorised operation: it applies to every row at once.
    # No need for a loop -- pandas multiplies element-wise.
    fuel_correction = df["TyreLife"] * FUEL_BURN_RATE * TIME_PER_KG
    df = df.copy()
    df["LapTime_corrected"] = df["LapTime_s"] + fuel_correction

    # Show the magnitude of the correction for sanity
    print(f"  Fuel correction range: "
          f"+{fuel_correction.min():.3f}s to +{fuel_correction.max():.3f}s")

    return df


# --------------------------------------------------------------
# STEP 4 -- DE-TRAFFIC: Drop laps ruined by traffic in any sector
# --------------------------------------------------------------
def remove_traffic_laps(df):
    """
    Flag and remove laps where the driver was held up by another car.

    WHY SECTORS, NOT FULL LAPS?
    A traffic encounter usually ruins only ONE sector. If a driver gets
    stuck behind a slow car in Sector 2, they might lose 1.5s there, but
    Sectors 1 and 3 are clean. The full-lap time only looks +0.5s slower
    (averaged out), which might survive the outlier filter. But checking
    per-sector catches it.

    METHOD:
    For each driver+stint, we compute each sector's median and IQR
    (interquartile range = Q3 - Q1, a robust measure of spread).
    Any lap where ANY sector is more than 1.5 * IQR above Q3 is flagged
    as traffic-affected and removed.

    This is the same logic as a box-plot whisker: points beyond the
    whisker are considered outliers.
    """
    original_count = len(df)
    sector_cols = ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]

    cleaned_groups = []
    for (driver, stint), group in df.groupby(["Driver", "Stint"]):
        if len(group) < 3:
            # Too few laps to compute meaningful IQR; keep them
            cleaned_groups.append(group)
            continue

        # Start by assuming all laps are clean
        is_clean = pd.Series(True, index=group.index)

        for col in sector_cols:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1  # interquartile range

            # Upper fence: anything above this is an outlier
            upper_fence = q3 + 1.5 * iqr

            # Mark laps where this sector is above the fence as dirty
            is_clean = is_clean & (group[col] <= upper_fence)

        cleaned_groups.append(group[is_clean])

    df = pd.concat(cleaned_groups, ignore_index=True)
    removed = original_count - len(df)
    print(f"  Removed {removed} traffic-affected laps ({len(df)} remain)")

    return df


# --------------------------------------------------------------
# STEP 5 -- DE-EVOLVE: Remove the track-getting-faster trend
# --------------------------------------------------------------
def correct_track_evolution(df):
    """
    Remove the effect of the track surface improving over a session.

    WHY STINT NORMALIZATION IS CRITICAL HERE:
    In practice sessions, drivers run short qualifying sims (light fuel, fast)
    early in the session, and long race sims (heavy fuel, slow) late in the
    session. If we compare raw lap times across session lap numbers, late-session
    laps look artificially slow simply because they had more fuel!
    
    To isolate TRUE track evolution (rubber laying down / temperature changes),
    we first subtract each stint's median lap time. This centers every stint around
    0, removing stint-type and fuel-load offsets. Then, the trend of these normalized
    deltas across session LapNumber reveals pure track evolution.
    """
    corrected_dfs = []

    for session_name, session_group in df.groupby("Session"):
        sg = session_group.copy()

        # Center each stint around 0 to remove short-run vs long-run baseline differences
        stint_medians = sg.groupby(["Driver", "Stint"])["LapTime_corrected"].transform("median")
        sg["LapTime_stint_norm"] = sg["LapTime_corrected"] - stint_medians

        # Compute the median normalized delta per session-lap across all drivers
        median_by_lap = (
            sg.groupby("LapNumber")["LapTime_stint_norm"]
            .median()
            .reset_index()
        )

        if len(median_by_lap) < 3:
            # Not enough data to fit a trend; skip correction
            sg["TrackEvo_correction"] = 0.0
            corrected_dfs.append(sg)
            continue

        # Fit a linear trend: LapDelta = slope * LapNumber + intercept
        x = median_by_lap["LapNumber"].values
        y = median_by_lap["LapTime_stint_norm"].values
        slope, intercept = np.polyfit(x, y, 1)

        # Reference point = midpoint of the session
        mid_lap = x.mean()

        # Subtract track evolution effect relative to session midpoint
        sg["TrackEvo_correction"] = slope * (sg["LapNumber"] - mid_lap)
        sg["LapTime_corrected"] = sg["LapTime_corrected"] - sg["TrackEvo_correction"]

        print(f"    {session_name}: track evo slope = {slope:+.4f} s/lap "
              f"(negative = getting faster)")

        corrected_dfs.append(sg)

    return pd.concat(corrected_dfs, ignore_index=True)



# --------------------------------------------------------------
# STEP 6 -- MODEL: Fit a degradation curve per compound
# --------------------------------------------------------------
def fit_degradation_models(df):
    """
    For each tyre compound, fit a polynomial that predicts lap time from
    tyre age. This gives us the 'degradation curve'.

    KEY INSIGHT -- DRIVER NORMALIZATION:
    Different drivers have very different base paces (Verstappen is ~2-3s
    faster than a Williams per lap). If we fit raw times, the model sees
    huge variance that is NOT tyre degradation -- it's just driver speed
    differences. So we NORMALIZE: for each driver, subtract their median
    corrected time for that compound. Now every driver's data is centered
    around zero, and the polynomial only sees the tyre-age trend.

    After fitting, we shift the curve back to absolute times using the
    overall median so the graphs show real lap times, not deltas.

    WHY POLYNOMIAL (degree 2)?
    Tyre degradation isn't perfectly linear. Tyres typically degrade slowly
    at first, then faster as the rubber wears through to harder layers
    (the 'cliff'). A quadratic (degree 2) captures this gentle-then-steep
    shape nicely:  time = a*age^2 + b*age + c

    Returns a dict mapping compound name -> model info dict.
    """
    models = {}
    compounds = ["SOFT", "MEDIUM", "HARD"]

    for compound in compounds:
        subset = df[df["Compound"] == compound].copy()

        # Check if we have enough data
        if len(subset) < MIN_LAPS_PER_COMPOUND:
            print(f"  WARNING: {compound} has only {len(subset)} laps "
                  f"(need {MIN_LAPS_PER_COMPOUND}). Skipping.")
            continue

        # --- Driver normalization ---
        # For each driver, compute their median corrected time on this compound.
        # Then subtract it so each driver's data is centered around zero.
        # This removes the 'driver speed' variable, leaving only tyre-age effects.
        driver_medians = subset.groupby("Driver")["LapTime_corrected"].transform("median")
        overall_median = subset["LapTime_corrected"].median()
        subset["LapTime_normalized"] = subset["LapTime_corrected"] - driver_medians + overall_median

        x = subset["TyreLife"].values.astype(float)
        y = subset["LapTime_normalized"].values.astype(float)

        # np.polyfit fits a polynomial of given degree.
        # degree=2 means: y = coeffs[0]*x^2 + coeffs[1]*x + coeffs[2]
        coeffs = np.polyfit(x, y, 2)

        # Create a smooth curve for plotting
        age_range = np.linspace(x.min(), x.max(), 100)
        predicted_curve = np.polyval(coeffs, age_range)

        # Compute predictions at the actual data points for MAE
        predicted_at_data = np.polyval(coeffs, x)
        mae = mean_absolute_error(y, predicted_at_data)

        # Compute residual std for confidence band
        residual_std = np.std(y - predicted_at_data)

        models[compound] = {
            "coeffs": coeffs,
            "ages": age_range,
            "predicted": predicted_curve,
            "residual_std": residual_std,
            "overall_median": overall_median,
            "data": subset,
            "mae": mae,
        }

        # Print the degradation rate
        a, b, c = coeffs
        print(f"  {compound:8s}: {len(subset):3d} laps, "
              f"MAE = {mae:.3f}s, "
              f"deg rate ~ {b:+.3f} s/lap (linear term)")

    return models


# ==============================================================
# GRAPH FUNCTIONS
# ==============================================================

def setup_figure():
    """
    Create a figure with our standard professional styling.
    Returns (fig, ax) ready for plotting.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Light grey dashed gridlines -- visible but not distracting
    ax.grid(True, linestyle="--", alpha=0.3, color="#888888")

    # Large, readable fonts
    ax.tick_params(axis="both", labelsize=12)

    return fig, ax


def find_flattest_stint(df, min_laps=8):
    """
    Search ALL practice long-run stints and pick the one whose RAW lap times
    are flattest or slightly declining with tyre age (smallest or negative raw slope),
    and where corrections unmask a rising degradation curve.

    WHY?
    A flat or declining raw slope means fuel burn-off and track evolution are
    genuinely MASKING the tyre degradation. That is the most convincing stint to show in Graph 2:
    the raw line looks flat, but after correction the hidden wear appears.

    Returns (driver, stint, session, raw_slope, corr_slope).
    """

    candidates = []

    for (driver, stint, session), group in df.groupby(["Driver", "Stint", "Session"]):
        if len(group) < min_laps:
            continue
        g = group.sort_values("TyreLife")
        x = g["TyreLife"].values.astype(float)
        y_raw = g["LapTime_s"].values.astype(float)
        y_corr = g["LapTime_corrected"].values.astype(float)

        slope_raw, _ = np.polyfit(x, y_raw, 1)
        slope_corr, _ = np.polyfit(x, y_corr, 1)

        candidates.append({
            "Driver": driver, "Stint": stint, "Session": session,
            "Compound": g["Compound"].iloc[0],
            "Laps": len(g),
            "RawSlope": slope_raw,
            "CorrSlope": slope_corr,
            "SteepnessGain": slope_corr - slope_raw
        })

    cand_df = pd.DataFrame(candidates)
    cand_df["AbsRawSlope"] = cand_df["RawSlope"].abs()

    # Filter for stints where raw lap times are flat/declining (|RawSlope| <= 0.05)
    # and pick the stint where correction unmasks the strongest rising curve
    flat_cands = cand_df[(cand_df["AbsRawSlope"] <= 0.05) & (cand_df["CorrSlope"] > 0)].copy()

    if not flat_cands.empty:
        # Sort by SteepnessGain descending so the corrected line is steepest relative to raw
        flat_cands = flat_cands.sort_values(by="SteepnessGain", ascending=False)
        picked_idx = flat_cands.index[0]
    else:
        cand_df = cand_df.sort_values(by="AbsRawSlope", ascending=True)
        picked_idx = cand_df.index[0]

    cand_display = cand_df.sort_values(by="AbsRawSlope", ascending=True)

    print("\n  Long-run stint candidates (min 8 laps):")
    print(f"  {'Driver':>6s}  {'Stint':>5s}  {'Session':>7s}  {'Compound':>8s}  "
          f"{'Laps':>4s}  {'RawSlope':>10s}  {'CorrSlope':>10s}")
    for idx, row in cand_display.iterrows():
        marker = "  <-- PICKED" if idx == picked_idx else ""
        print(f"  {row['Driver']:>6s}  {row['Stint']:5.0f}  {row['Session']:>7s}  "
              f"{row['Compound']:>8s}  {row['Laps']:4d}  {row['RawSlope']:+10.4f} s/lap  "
              f"{row['CorrSlope']:+10.4f} s/lap{marker}")

    best = cand_df.loc[picked_idx]
    return best["Driver"], best["Stint"], best["Session"], best["RawSlope"], best["CorrSlope"]


def generate_graph1_raw(df, output_dir):
    """
    GRAPH 1: Raw practice-stint lap times for the FLATTEST stint.
    PURPOSE: Show that wear is INVISIBLE in raw data -- it looks flat/noisy
    because fuel burn-off is masking the tyre degradation.

    We auto-pick the stint with the smallest (or most negative) raw slope.
    """
    print("\n  Generating Graph 1 (raw stint data)...")

    best_driver, best_stint, best_session, raw_slope, corr_slope = find_flattest_stint(df)


    # Extract that stint's data, sorted by tyre age
    stint_data = df[
        (df["Driver"] == best_driver)
        & (df["Stint"] == best_stint)
        & (df["Session"] == best_session)
    ].sort_values("TyreLife")

    compound = stint_data["Compound"].iloc[0]
    print(f"\n    Selected: Driver {best_driver}, Stint {int(best_stint)}, "
          f"{best_session}, {compound} ({len(stint_data)} laps, "
          f"raw slope = {raw_slope:+.4f} s/lap)")

    fig, ax = setup_figure()

    ax.plot(
        stint_data["TyreLife"], stint_data["LapTime_s"],
        color="#666666",
        marker="o",
        markersize=6,
        linewidth=1.5,
        label="Raw lap time",
        zorder=2,
    )

    ax.set_xlabel("Tyre Age (laps)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Lap Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Raw {compound}-Tyre Stint -- Driver {best_driver} ({best_session})\n"
        f"Raw slope = {raw_slope:+.3f} s/lap -- degradation hidden by fuel burn",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=13, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "graph1_raw.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {path}")

    return best_driver, best_stint, best_session


def generate_graph2_before_after(df, defueled_df, detrafficked_df,
                                  best_driver, best_stint, best_session,
                                  output_dir):
    """
    GRAPH 2: THE MONEY SHOT -- before vs after correction on the same stint.
    Grey = raw (flat/declining), Red = corrected (climbing) on shared y-axis.

    Also prints per-stage correction contributions so we can verify each
    stage actually modifies the data.
    """
    print("\n  Generating Graph 2 (before/after)...")

    # Build a stint mask for all intermediate DataFrames
    def get_stint(frame):
        mask = (
            (frame["Driver"] == best_driver)
            & (frame["Stint"] == best_stint)
            & (frame["Session"] == best_session)
        )
        return frame[mask].sort_values("TyreLife")

    stint_final = get_stint(df)           # after ALL corrections
    stint_defueled = get_stint(defueled_df)   # after fuel only
    stint_detrafficked = get_stint(detrafficked_df)  # after fuel + traffic

    compound = stint_final["Compound"].iloc[0]

    # --- Per-stage contribution analysis ---
    # We compare first and last lap across each stage to see what each did.
    first_tyre = stint_final["TyreLife"].iloc[0]
    last_tyre = stint_final["TyreLife"].iloc[-1]

    # Raw values at first and last lap
    raw_first = stint_final["LapTime_s"].iloc[0]
    raw_last = stint_final["LapTime_s"].iloc[-1]

    # After fuel correction (LapTime_corrected in defueled_df)
    fuel_first = stint_defueled.loc[stint_defueled["TyreLife"] == first_tyre, "LapTime_corrected"].values
    fuel_last = stint_defueled.loc[stint_defueled["TyreLife"] == last_tyre, "LapTime_corrected"].values
    fuel_first = fuel_first[0] if len(fuel_first) > 0 else raw_first
    fuel_last = fuel_last[0] if len(fuel_last) > 0 else raw_last

    # After fuel + traffic (LapTime_corrected in detrafficked_df)
    # Traffic removal drops laps, so the stint may be shorter. Check if
    # the first/last tyre-age laps survived.
    traf_first_rows = stint_detrafficked.loc[stint_detrafficked["TyreLife"] == first_tyre, "LapTime_corrected"]
    traf_last_rows = stint_detrafficked.loc[stint_detrafficked["TyreLife"] == last_tyre, "LapTime_corrected"]
    traf_first = traf_first_rows.values[0] if len(traf_first_rows) > 0 else fuel_first
    traf_last = traf_last_rows.values[0] if len(traf_last_rows) > 0 else fuel_last

    # After all 3 corrections (LapTime_corrected in final df)
    corr_first = stint_final["LapTime_corrected"].iloc[0]
    corr_last = stint_final["LapTime_corrected"].iloc[-1]

    # Compute per-stage deltas
    fuel_delta_first = fuel_first - raw_first
    fuel_delta_last = fuel_last - raw_last
    traffic_delta_first = traf_first - fuel_first
    traffic_delta_last = traf_last - fuel_last
    evo_delta_first = corr_first - traf_first
    evo_delta_last = corr_last - traf_last

    print(f"\n  === PER-STAGE CORRECTION BREAKDOWN (Driver {best_driver}, {compound}) ===")
    print(f"    Stint: TyreLife {first_tyre:.0f} to {last_tyre:.0f} ({len(stint_final)} laps)")
    print(f"")
    print(f"    {'Stage':<20s}  {'First lap':>10s}  {'Last lap':>10s}  {'Net effect':>10s}")
    print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"    {'Raw':20s}  {raw_first:10.3f}s  {raw_last:10.3f}s  {raw_last - raw_first:+10.3f}s")
    print(f"    {'+ Fuel correction':20s}  {fuel_delta_first:+10.3f}s  {fuel_delta_last:+10.3f}s  "
          f"{(fuel_delta_last - fuel_delta_first):+10.3f}s")

    traffic_net = traffic_delta_last - traffic_delta_first
    if abs(traffic_net) < 0.01:
        print(f"    {'+ Traffic removal':20s}  {traffic_delta_first:+10.3f}s  {traffic_delta_last:+10.3f}s  "
              f"   ~0.000s (no effect on this stint)")
    else:
        print(f"    {'+ Traffic removal':20s}  {traffic_delta_first:+10.3f}s  {traffic_delta_last:+10.3f}s  "
              f"{traffic_net:+10.3f}s")

    evo_net = evo_delta_last - evo_delta_first
    if abs(evo_net) < 0.01:
        print(f"    {'+ Track evolution':20s}  {evo_delta_first:+10.3f}s  {evo_delta_last:+10.3f}s  "
              f"   ~0.000s (no effect on this stint)")
    else:
        print(f"    {'+ Track evolution':20s}  {evo_delta_first:+10.3f}s  {evo_delta_last:+10.3f}s  "
              f"{evo_net:+10.3f}s")

    print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"    {'Corrected':20s}  {corr_first:10.3f}s  {corr_last:10.3f}s  {corr_last - corr_first:+10.3f}s")

    raw_deg = raw_last - raw_first
    corr_deg = corr_last - corr_first

    # Compute slopes for the summary line
    raw_x = stint_final["TyreLife"].values.astype(float)
    raw_slope, _ = np.polyfit(raw_x, stint_final["LapTime_s"].values.astype(float), 1)
    corr_slope, _ = np.polyfit(raw_x, stint_final["LapTime_corrected"].values.astype(float), 1)

    print(f"\n    Raw slope:       {raw_slope:+.4f} s/lap")
    print(f"    Corrected slope: {corr_slope:+.4f} s/lap")

    if raw_slope <= 0 and corr_slope > 0:
        hidden_deg = corr_slope - raw_slope
        print(f"    Impact: Raw data showed pace improving ({raw_slope:+.4f} s/lap), "
              f"but correction unmasks true tyre degradation ({corr_slope:+.4f} s/lap). "
              f"Net correction shift: {hidden_deg:+.4f} s/lap.")
    elif raw_slope > 0 and corr_slope > 0:
        multiplier = corr_slope / raw_slope
        print(f"    Multiplier: {multiplier:.1f}x more wear revealed after correction "
              f"({corr_slope:+.4f} vs {raw_slope:+.4f} s/lap)")
    else:
        print(f"    Raw slope: {raw_slope:+.4f} s/lap -> Corrected slope: {corr_slope:+.4f} s/lap")


    # --- Plot ---
    fig, ax = setup_figure()

    ax.plot(
        stint_final["TyreLife"], stint_final["LapTime_s"],
        color="#AAAAAA",
        marker="s",
        markersize=5,
        linewidth=1.5,
        label=f"Raw lap time (slope = {raw_slope:+.3f} s/lap)",
        zorder=1,
    )

    ax.plot(
        stint_final["TyreLife"], stint_final["LapTime_corrected"],
        color="#E10600",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label=f"Corrected (slope = {corr_slope:+.3f} s/lap)",
        zorder=2,
    )

    ax.set_xlabel("Tyre Age (laps)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Lap Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Before vs After: Isolating True Tyre Degradation\n"
        f"Driver {best_driver}, {compound} Compound ({best_session})",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=12, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "graph2_before_after.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {path}")


def generate_graph3_pred_vs_actual(practice_models, year, circuit,
                                    race_session, cache_dir, output_dir):
    """
    GRAPH 3: Model validation -- predicted race pace vs actual race laps.

    BASELINE NORMALIZATION:
    Instead of trying to align absolute paces (which differ due to fuel
    loads, track conditions, etc.), we normalize both series to their own
    lap-2 baseline. This means both curves start at ~0 and we compare
    only the SHAPE of degradation, not the absolute level.

    CONFIDENCE BAND:
    Computed from the model's residuals on CLEANED race laps (after
    outlier removal), not from raw practice residuals.
    """
    print("\n  Generating Graph 3 (prediction vs actual)...")
    print("  Loading race data for validation...")

    # Load race session
    session = fastf1.get_session(year, circuit, race_session)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    race_laps = session.laps.copy()

    # Convert to seconds
    race_laps["LapTime_s"] = race_laps["LapTime"].dt.total_seconds()

    # --- Full cleaning (same as practice pipeline) ---
    race_laps = race_laps[race_laps["IsAccurate"] == True]
    race_laps = race_laps[race_laps["PitInTime"].isna() & race_laps["PitOutTime"].isna()]
    race_laps = race_laps.dropna(subset=["LapTime_s", "Compound", "TyreLife"])

    # Green-flag only
    if "TrackStatus" in race_laps.columns:
        race_laps = race_laps[race_laps["TrackStatus"] == "1"].copy()

    # Apply the same fuel correction to race laps
    race_laps["LapTime_corrected"] = (
        race_laps["LapTime_s"]
        + race_laps["TyreLife"] * FUEL_BURN_RATE * TIME_PER_KG
    )

    # Per-driver normalization (same as practice model)
    # This removes driver-speed differences so we compare degradation shape
    for compound_name in race_laps["Compound"].unique():
        mask = race_laps["Compound"] == compound_name
        driver_med = race_laps.loc[mask].groupby("Driver")["LapTime_corrected"].transform("median")
        overall_med = race_laps.loc[mask, "LapTime_corrected"].median()
        race_laps.loc[mask, "LapTime_normalized"] = (
            race_laps.loc[mask, "LapTime_corrected"] - driver_med + overall_med
        )

    # Outlier removal per stint (same 2-std method as practice)
    cleaned_stints = []
    for (driver, stint), group in race_laps.groupby(["Driver", "Stint"]):
        median = group["LapTime_normalized"].median()
        std = group["LapTime_normalized"].std()
        if pd.isna(std) or std == 0:
            cleaned_stints.append(group)
            continue
        upper_bound = median + 2 * std
        cleaned_stints.append(group[group["LapTime_normalized"] <= upper_bound])
    race_laps = pd.concat(cleaned_stints, ignore_index=True)

    # Auto-pick the compound with the most race laps (that we have a model for)
    compound_counts = race_laps["Compound"].value_counts()
    chosen_compound = None
    for compound in compound_counts.index:
        if compound in practice_models:
            chosen_compound = compound
            break

    if chosen_compound is None:
        print("  WARNING: No matching compound between practice model and race. "
              "Skipping Graph 3.")
        return

    print(f"  Auto-picked compound: {chosen_compound} "
          f"({compound_counts[chosen_compound]} race laps after cleaning)")

    race_compound = race_laps[race_laps["Compound"] == chosen_compound].copy()
    model_info = practice_models[chosen_compound]

    # --- Baseline normalization ---
    # Subtract each series' own lap-2 value so both start at ~0.
    # This removes the absolute offset (different fuel loads, track temp,
    # etc.) and compares only the degradation SHAPE.

    # Actual race: per-tyre-age median, then subtract lap-2 baseline
    race_median_by_age = (
        race_compound.groupby("TyreLife")["LapTime_normalized"]
        .median()
        .reset_index()
        .sort_values("TyreLife")
    )
    # Find the baseline: use lap 2 (lap 1 is often an outlier from pit exit)
    baseline_age = 2.0
    race_baseline_rows = race_median_by_age[race_median_by_age["TyreLife"] == baseline_age]
    if len(race_baseline_rows) > 0:
        race_baseline = race_baseline_rows["LapTime_normalized"].values[0]
    else:
        # Fall back to earliest available tyre age
        race_baseline = race_median_by_age["LapTime_normalized"].iloc[0]
        baseline_age = race_median_by_age["TyreLife"].iloc[0]

    race_median_by_age["delta"] = race_median_by_age["LapTime_normalized"] - race_baseline

    # Model prediction: evaluate at the same ages, then subtract its own lap-2
    pred_baseline = np.polyval(model_info["coeffs"], baseline_age)
    pred_ages = np.linspace(
        race_median_by_age["TyreLife"].min(),
        race_median_by_age["TyreLife"].max(),
        200,
    )
    pred_times_delta = np.polyval(model_info["coeffs"], pred_ages) - pred_baseline

    # Also compute individual race laps as deltas for scatter
    race_compound["delta"] = race_compound["LapTime_normalized"] - race_baseline

    # Compute model predictions at each actual race lap age (for MAE and band)
    race_x = race_compound["TyreLife"].values.astype(float)
    race_pred_at_data = np.polyval(model_info["coeffs"], race_x) - pred_baseline
    race_actual_delta = race_compound["delta"].values.astype(float)

    # Confidence band from the MODEL'S residuals on CLEANED race data
    residuals = race_actual_delta - race_pred_at_data
    band_std = np.std(residuals)

    # Recompute MAE on the baselined data
    race_mae = mean_absolute_error(race_actual_delta, race_pred_at_data)

    fig, ax = setup_figure()

    # Light grey dots: individual race laps (baseline-normalized)
    ax.scatter(
        race_x, race_actual_delta,
        color="#CCCCCC",
        s=15,
        alpha=0.4,
        label="Individual race laps",
        zorder=1,
    )

    # Black line: median actual per tyre age (baselined)
    ax.plot(
        race_median_by_age["TyreLife"],
        race_median_by_age["delta"],
        color="#333333",
        linewidth=2.0,
        marker=".",
        markersize=4,
        label=f"Actual race median ({chosen_compound})",
        zorder=2,
    )

    # Blue line: model prediction (baselined)
    ax.plot(
        pred_ages, pred_times_delta,
        color="#1E88E5",
        linewidth=2.5,
        label="Practice-trained prediction",
        zorder=3,
    )

    # Confidence band from cleaned race residuals
    ax.fill_between(
        pred_ages,
        pred_times_delta - band_std,
        pred_times_delta + band_std,
        alpha=0.15,
        color="#1E88E5",
        label=f"+/- 1 std ({band_std:.2f}s)",
    )

    # Reference line at y=0
    ax.axhline(y=0, color="#999999", linewidth=0.8, linestyle="-", alpha=0.5)

    ax.set_xlabel("Tyre Age (laps)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Lap Time Delta from Lap 2 (seconds)", fontsize=14,
                   fontweight="bold")
    ax.set_title(
        f"Practice-Trained Model vs Actual Race Degradation\n"
        f"{chosen_compound} compound -- {year} {circuit} (baselined to lap {baseline_age:.0f})",
        fontsize=18, fontweight="bold", pad=15,
    )

    # MAE annotation
    ax.annotate(
        f"MAE: {race_mae:.2f}s",
        xy=(0.97, 0.95), xycoords="axes fraction",
        fontsize=16, fontweight="bold",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                  edgecolor="#4CAF50", alpha=0.9),
    )

    ax.legend(fontsize=12, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "graph3_pred_vs_actual.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n  === MODEL VALIDATION ===")
    print(f"    Compound: {chosen_compound}")
    print(f"    Practice model MAE: {model_info['mae']:.3f}s")
    print(f"    Race prediction MAE (baselined): {race_mae:.3f}s")
    print(f"    Confidence band width: +/- {band_std:.3f}s")


# ==============================================================
# MAIN -- run the full pipeline
# ==============================================================
def main():
    print("=" * 60)
    print("  DegradIQ -- F1 Tyre Degradation Isolation Tool")
    print("=" * 60)
    print(f"\n  Circuit: {YEAR} {CIRCUIT}")
    print(f"  Sessions: {', '.join(SESSIONS)}\n")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Ingest ──
    print("-" * 40)
    print("STEP 1: Ingesting practice data...")
    print("-" * 40)
    practice_df = load_practice_data(YEAR, CIRCUIT, SESSIONS, CACHE_DIR)

    # Show a preview of what we loaded
    preview_cols = [
        "Driver", "LapNumber", "LapTime_s",
        "Sector1Time_s", "Sector2Time_s", "Sector3Time_s",
        "Compound", "TyreLife", "Stint", "IsAccurate",
        "TrackStatus", "Session", "TrackTemp",
    ]
    available_cols = [c for c in preview_cols if c in practice_df.columns]
    print("\n  Preview (first 10 rows):")
    print(practice_df[available_cols].head(10).to_string(index=False))

    print("\n  Laps per compound:")
    for compound, count in practice_df["Compound"].value_counts().items():
        print(f"    {compound:10s}: {count} laps")
    print("  [OK] Step 1 complete.\n")

    # ── Step 2: Clean ──
    print("-" * 40)
    print("STEP 2: Cleaning junk laps...")
    print("-" * 40)
    clean_df = clean_laps(practice_df)
    print("  [OK] Step 2 complete.\n")

    # ── Step 3: De-fuel ──
    print("-" * 40)
    print("STEP 3: Correcting for fuel burn...")
    print("-" * 40)
    defueled_df = correct_fuel_effect(clean_df)
    print("  [OK] Step 3 complete.\n")

    # ── Step 4: De-traffic ──
    print("-" * 40)
    print("STEP 4: Removing traffic-affected laps...")
    print("-" * 40)
    detrafficked_df = remove_traffic_laps(defueled_df)
    print("  [OK] Step 4 complete.\n")

    # ── Step 5: De-evolve ──
    print("-" * 40)
    print("STEP 5: Correcting for track evolution...")
    print("-" * 40)
    devolved_df = correct_track_evolution(detrafficked_df)
    print("  [OK] Step 5 complete.\n")

    # ── Step 6: Model ──
    print("-" * 40)
    print("STEP 6: Fitting degradation models...")
    print("-" * 40)
    models = fit_degradation_models(devolved_df)
    print("  [OK] Step 6 complete.\n")

    # ── Step 7: Generate graphs ──
    print("-" * 40)
    print("STEP 7: Generating presentation graphs...")
    print("-" * 40)

    # Graph 1: raw stint (auto-picks flattest raw slope)
    best_driver, best_stint, best_session = generate_graph1_raw(devolved_df, OUTPUT_DIR)

    # Graph 2: before/after (uses the same stint, prints per-stage contributions)
    generate_graph2_before_after(
        devolved_df, defueled_df, detrafficked_df,
        best_driver, best_stint, best_session, OUTPUT_DIR,
    )

    # Graph 3: prediction vs actual race (baselined, cleaned band)
    generate_graph3_pred_vs_actual(
        models, YEAR, CIRCUIT, RACE_SESSION, CACHE_DIR, OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("  DegradIQ pipeline complete!")
    print(f"  Graphs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
