"""
DegradIQ — Final Reproducibility Benchmark
============================================
Run this script twice in a row.  The numbers MUST be identical.

Usage:
    python src/benchmark.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import io
import contextlib

# ── Fixed seed — must appear before ANY other import that touches randomness ──
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import degradiq  # noqa: E402 (seed must be set first)

CIRCUITS    = ["Bahrain", "Spain", "Italy"]
YEAR        = 2024
SESSIONS    = ["FP1", "FP2", "FP3"]
CACHE_DIR   = "./data"
OUTPUT_DIR  = "./outputs/figures"


def run_circuit(circuit):
    """
    Run the full pipeline for one circuit and return all key numbers as a dict.
    All intermediate prints are suppressed.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        practice_df     = degradiq.load_practice_data(YEAR, circuit, SESSIONS, CACHE_DIR)
        filter_fn       = getattr(degradiq, "filter_long_runs", lambda df: df)
        longrun_df      = filter_fn(practice_df)
        clean_df        = degradiq.clean_laps(longrun_df)
        defueled_df     = degradiq.correct_fuel_effect(clean_df)
        detrafficked_df = degradiq.remove_traffic_laps(defueled_df)
        devolved_df     = degradiq.correct_track_evolution(detrafficked_df)
        models          = degradiq.fit_degradation_models(devolved_df)
        best_driver, best_stint, best_session = degradiq.generate_graph1_raw(
            devolved_df, OUTPUT_DIR
        )

    # ── Chosen stint data ──────────────────────────────────────────────────
    stint_final = devolved_df[
        (devolved_df["Driver"]  == best_driver)  &
        (devolved_df["Stint"]   == best_stint)   &
        (devolved_df["Session"] == best_session)
    ].sort_values("TyreLife")

    compound   = stint_final["Compound"].iloc[0]
    n_laps     = len(stint_final)
    first_tyre = int(stint_final["TyreLife"].iloc[0])
    last_tyre  = int(stint_final["TyreLife"].iloc[-1])

    raw_x         = stint_final["TyreLife"].values.astype(float)
    raw_slope, _  = np.polyfit(raw_x, stint_final["LapTime_s"].values.astype(float), 1)
    corr_slope, _ = np.polyfit(raw_x, stint_final["LapTime_corrected"].values.astype(float), 1)

    below_threshold    = corr_slope < 0.0
    corr_slope_clamped = max(corr_slope, 0.0)

    if below_threshold:
        multiplier_str = "N/A (below threshold)"
    elif raw_slope > 0 and corr_slope_clamped > 0:
        multiplier_str = f"{corr_slope_clamped / raw_slope:.1f}x"
    else:
        multiplier_str = "N/A"

    # ── Noise attribution stage values ─────────────────────────────────────
    # Read each pipeline-stage's corrected value for this specific stint
    def get_stint(df):
        return df[
            (df["Driver"]  == best_driver)  &
            (df["Stint"]   == best_stint)   &
            (df["Session"] == best_session)
        ].sort_values("TyreLife")

    stint_defueled     = get_stint(defueled_df)
    stint_detrafficked = get_stint(detrafficked_df)

    raw_first  = stint_final["LapTime_s"].iloc[0]
    raw_last   = stint_final["LapTime_s"].iloc[-1]

    fuel_first = stint_defueled["LapTime_corrected"].iloc[0]
    fuel_last  = stint_defueled["LapTime_corrected"].iloc[-1]

    traf_first = stint_detrafficked["LapTime_corrected"].iloc[0] if len(stint_detrafficked) else fuel_first
    traf_last  = stint_detrafficked["LapTime_corrected"].iloc[-1] if len(stint_detrafficked) else fuel_last

    corr_first = stint_final["LapTime_corrected"].iloc[0]
    corr_last  = stint_final["LapTime_corrected"].iloc[-1]

    fuel_delta_first  = fuel_first - raw_first
    fuel_delta_last   = fuel_last  - raw_last
    fuel_contribution = fuel_delta_last - fuel_delta_first

    evo_contribution  = (corr_last - traf_last) - (corr_first - traf_first)

    true_wear_raw = corr_last - corr_first
    true_wear     = max(true_wear_raw, 0.0)

    defueled_ages     = set(stint_defueled["TyreLife"].values)
    detrafficked_ages = set(stint_detrafficked["TyreLife"].values)
    removed_ages      = defueled_ages - detrafficked_ages
    n_traffic_laps    = len(removed_ages)
    if n_traffic_laps > 0:
        removed_times   = stint_defueled[stint_defueled["TyreLife"].isin(removed_ages)]["LapTime_corrected"]
        clean_median    = stint_detrafficked["LapTime_corrected"].median()
        avg_traffic_dev = (removed_times - clean_median).mean()
    else:
        avg_traffic_dev = 0.0

    # ── Validation MAE from Graph 3 ────────────────────────────────────────
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        degradiq.generate_graph3_pred_vs_actual(
            models, YEAR, circuit, "R", CACHE_DIR, OUTPUT_DIR
        )
    val_mae = None
    for line in buf3.getvalue().splitlines():
        if "Race prediction MAE" in line:
            try:
                val_mae = float(line.split(":")[-1].strip().rstrip("s"))
            except ValueError:
                pass

    return {
        "circuit":          circuit,
        "driver":           best_driver,
        "stint":            int(best_stint),
        "session":          best_session,
        "compound":         compound,
        "n_laps":           n_laps,
        "tyre_range":       f"{first_tyre}-{last_tyre}",
        "raw_slope":        raw_slope,
        "corr_slope":       corr_slope,
        "corr_clamped":     corr_slope_clamped,
        "below_threshold":  below_threshold,
        "multiplier_str":   multiplier_str,
        "val_mae":          val_mae,
        "fuel_contribution": fuel_contribution,
        "evo_contribution":  evo_contribution,
        "n_traffic_laps":    n_traffic_laps,
        "avg_traffic_dev":   avg_traffic_dev,
        "true_wear_raw":     true_wear_raw,
        "true_wear":         true_wear,
    }


def print_results(results, run_num):
    """Print the final summary table and per-circuit attribution blocks."""
    SEP = "=" * 76

    print()
    print(SEP)
    print(f"  DEGRADIQ FINAL BENCHMARK  (RUN #{run_num})  seed={RANDOM_SEED}")
    print(SEP)
    print()

    # Table header
    print(f"  {'Circuit':<9} | {'Stint':<20} | {'Raw slope':>10} | {'Corrected slope':>15} | {'Multiplier':>21} | {'Val MAE':>8}")
    print("  " + "-" * 92)

    for r in results:
        corr_disp   = "~0.00 s/lap *" if r["below_threshold"] else f"{r['corr_clamped']:+.4f} s/lap"
        mae_disp    = f"{r['val_mae']:.2f}s" if r["val_mae"] is not None else "N/A"
        stint_label = f"{r['driver']} {r['session']} {r['compound']} {r['n_laps']}L"
        print(f"  {r['circuit']:<9} | {stint_label:<20} | "
              f"{r['raw_slope']:+.4f} s/lap | {corr_disp:>15} | "
              f"{r['multiplier_str']:>21} | {mae_disp:>8}")

    if any(r["below_threshold"] for r in results):
        print()
        print("  * = corrected slope clamped to 0 (low-degradation circuit)")

    # Attribution blocks
    for r in results:
        print()
        print(f"  {'='*74}")
        print(f"  NOISE ATTRIBUTION  |  {r['circuit'].upper()}")
        print(f"  Stint: {r['driver']}, {r['compound']}, {r['session']}, "
              f"{r['n_laps']} laps (TyreLife {r['tyre_range']})")
        print(f"  {'-'*74}")
        print(f"    Fuel effect:        {r['fuel_contribution']:+.3f} s")
        print(f"    Track evolution:    {r['evo_contribution']:+.3f} s")
        if r["n_traffic_laps"] > 0:
            print(f"    Traffic removed:    {r['n_traffic_laps']} lap(s)  "
                  f"(avg {r['avg_traffic_dev']:+.3f}s off clean pace)")
        else:
            print(f"    Traffic removed:    0 laps flagged in this stint")
        print(f"  {'-'*74}")
        if r["below_threshold"]:
            print(f"    True tyre wear:     LOW-DEGRADATION CIRCUIT")
            print(f"                        Wear below reliable isolation threshold.")
            print(f"                        Raw corrected change = {r['true_wear_raw']:+.3f} s (noise floor)")
        else:
            print(f"    True tyre wear:     {r['true_wear']:+.3f} s")
            if r["true_wear"] > 0.01:
                fuel_pct = abs(r["fuel_contribution"]) / r["true_wear"] * 100
                evo_pct  = abs(r["evo_contribution"])  / r["true_wear"] * 100
                print(f"      Fuel hid:         {fuel_pct:.0f}% of the real wear")
                print(f"      Track evo hid:    {evo_pct:.0f}% of the real wear")
        print(f"  {'='*74}")

    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for run_num in [1, 2]:
        # Re-seed before each run so the two runs are truly independent
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        results = []
        for circuit in CIRCUITS:
            sys.stderr.write(f"  [Run {run_num}] Processing {circuit}...\n")
            sys.stderr.flush()
            results.append(run_circuit(circuit))
        print_results(results, run_num)
        all_results.append(results)

    # ── Reproducibility check ──────────────────────────────────────────────
    print("=" * 76)
    print("  REPRODUCIBILITY CHECK")
    print("=" * 76)
    keys_to_check = ["raw_slope", "corr_slope", "corr_clamped", "val_mae",
                     "fuel_contribution", "evo_contribution", "true_wear"]
    all_match = True
    for r1, r2 in zip(all_results[0], all_results[1]):
        for k in keys_to_check:
            v1, v2 = r1[k], r2[k]
            if v1 is None and v2 is None:
                continue
            if v1 is None or v2 is None or abs(v1 - v2) > 1e-9:
                print(f"  MISMATCH  {r1['circuit']}  {k}: {v1} vs {v2}")
                all_match = False
    if all_match:
        print()
        print("  ALL NUMBERS IDENTICAL ACROSS BOTH RUNS — pipeline is fully reproducible.")
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
