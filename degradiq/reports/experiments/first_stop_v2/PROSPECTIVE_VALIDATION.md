# Prospective validation protocol — first-stop model

A pre-registered protocol for a scientifically defensible, out-of-sample test. It is a
**protocol, not a result** — no prospective validation has been run. Its purpose is to
let DegradIQ earn a genuine blind-test claim that the historical 2024/2025 results
cannot support (2024 = development period; cached 2025 = previously examined; the
remaining 2025 lap-timing is blocked by FastF1 HTTP 403).

## Why this is needed
- Historical 2024 metrics are **development-period** results.
- Cached 2025 events were **examined** during development — not a blind holdout.
- The 13 remaining 2025 events could not be recovered (HTTP 403); see `BLIND_TEST_2025.md`.
So the only path to a real blind test is **prospective**: predict future races and
freeze the predictions before the outcomes exist.

## Frozen artefacts (register before any prediction)
| Item | Value |
|---|---|
| Model file | `degradiq/artifacts/first_stop_model_v2_hgb.joblib` |
| Model SHA-256 | `724ab2eb3368e65b2b9239191c1b026923d5b62c04189cd1be9e4c8359b3be38` |
| Architecture | calibrated HistGradientBoosting (sigmoid, GroupKFold-by-EventName calibration) |
| Training seasons | 2022–2024 |
| Label version | `2.0-strategic-green-vsc-sc` |
| Feature schema | the 11 `FEATURE_COLUMNS` in `degradiq/pit_strategy.py` |
| Library pins | `provenance.json` (python 3.13, sklearn 1.9.1, numpy 2.5.4, pandas 2.3.3) |

The served model is frozen for the whole protocol. **Do not retrain, recalibrate or
re-promote** mid-protocol; a change starts a new protocol version.

## Prediction checkpoints (per eligible race)
Produce and **immutably store** predictions at these points, each timestamped (UTC),
before the corresponding information exists:
1. **Pre-race** (after qualifying, before lights-out): grid, starting compound, team —
   `mode="pre-race"`. This is the headline prospective prediction.
2. **Point-in-time, end of lap k** (optional, during the race): only features available
   at the end of lap k — `mode="live"`/replay semantics. Never use any lap > k.

For each checkpoint record: season, round, `EventName`, driver code, prediction UTC
timestamp, model SHA-256, per-lap hazards, cumulative probability, predicted stop lap,
and the 10–90 interval.

## Outcome collection (after the race)
Record each driver's **actual first strategic stop** using the committed label logic
(`build_hazard_rows`, same exclusions: drive/pass-throughs, confirmed red-flag entries,
lap ≤2 SC/red compulsory). Censor drivers with no observed first stop.

## Eligibility, exclusions, missing-data policy
- **Eligible:** full-season drivers who start the race.
- **Exclude:** DNS; retirement before any stop (censored, not an error); races with
  missing/incomplete pit-in timing (recorded as unavailable, never imputed).
- **Missing features** at prediction time are imputed by the model pipeline only
  (median/most-frequent); never backfilled from post-race data.

## Baselines (same driver set, same checkpoint)
- Historical-median stop lap by (circuit, compound) with compound/global fallback.
- Marginal per-lap hazard.
- Empirical circuit/compound hazard.

## Metrics (classification and timing reported separately)
- **Classification:** Brier, log-loss, ROC-AUC over eligible per-lap rows.
- **Timing:** stop-lap MAE vs baselines (conditional on an observed stop).
- **Uncertainty:** 10–90 interval **coverage AND width** (state the 80% nominal).
- **Calibration:** reliability bins + ECE, plus quantile-binned reliability.
- Report per-event and aggregate (micro + macro). Never present classification skill as
  timing skill.

## Integrity rules
- Store predictions in an **append-only** record committed **before** each race; never
  revise a prediction after the outcome is known.
- The model SHA-256 in each stored prediction must equal the frozen value; a mismatch
  invalidates that race.
- Analyse only after a pre-declared number of races have completed; do not peek-and-stop.

## Execution
Use the **next eligible future rounds** as the prospective holdout. If the current
season has no remaining eligible rounds, register the protocol for the next season
(note: the 2026 driver lineup differs — `config.py` `FULL_SEASON_DRIVERS`). The
committed engine already supports pre-race and point-in-time prediction; the only new
component needed is an append-only prediction log + a post-race scoring step that reuses
`scripts/validate_generalization.py` metrics. **No prospective result may be reported
until outcomes are collected under these rules.**
