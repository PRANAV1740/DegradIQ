# Phase III — unseen-circuit timing: analysis and model comparison

Goal: improve first-stop *timing* on unfamiliar circuits, where the served
calibrated-HGB hazard model beats the historical-median baseline on only 5/25
circuits and its 10-90 interval is ~23 laps wide.

Reproduce: `python -m degradiq.scripts.timing_model_comparison` (offline). Full numbers
in [`timing_comparison_summary.json`](timing_comparison_summary.json).

## Where the current model fails
The worst unseen circuits are the ones with **tight, predictable** stop windows —
Hungary, Dutch, Las Vegas, China (baseline MAE 2.5–4.2 laps). There the hazard model's
survival-derived median is too diffuse and loses to a simple historical median by ~3
laps. So the weakness is **not** classification (which generalises) but an **over-wide,
poorly-localised timing distribution** on predictable circuits.

## Fair comparison (same certified data, LOCO + temporal, same baseline)
Candidate: a **direct quantile stop-lap regressor** — gradient boosting with pinball
loss at the 10th/50th/90th percentiles, predicting the stop lap from pre-race features
(race laps, grid, starting tyre age, compound, team, circuit). Conditional on an
observed stop (censoring ignored), the same convention as the hazard model's stop-lap MAE.

| Unseen-circuit (LOCO) | MAE | Coverage (nom. 0.80) | Interval width | Beats baseline |
|---|---|---|---|---|
| hazard_hgb (served) | 6.93 | 0.85 (over-covered) | 23.3 laps | 5/25 |
| quantile_gbr | 6.95 | 0.64 (under-covered) | **16.4 laps** | **11/25** |
| quantile_gbr − EventName | 7.05 | 0.68 | 17.2 laps | 11/25 |

Temporal 2024 (unseen season, seen circuit): quantile_gbr MAE **6.50** vs baseline 6.98
(and vs HGB 6.96).

## What the evidence says
1. **Sharper, more useful intervals are achievable:** the quantile regressor roughly
   halves the interval width (16 vs 23 laps) and beats the baseline on **11/25**
   circuits (vs 5/25) — a real improvement in timing *usefulness*.
2. **But neither model is well-calibrated:** the HGB is over-covered (0.85, too wide)
   and the quantile regressor is under-covered (0.64, too narrow). Sharper came at the
   cost of coverage.
3. **Unseen-circuit MAE is fundamentally capped** by the available features: the two
   models tie (~6.9), and removing `EventName` barely changes LOCO results (6.95 → 7.05)
   — because an unseen circuit's one-hot is all-zeros anyway, circuit identity cannot
   help there. The real lever is **circuit-descriptive / weather / traffic features**,
   which are not in the certified dataset and cannot be re-derived for the uncached
   events (FastF1 timing host returns 403; see `BLIND_TEST_2025.md`).

## Decision (no promotion)
The quantile regressor does **not** clear the "sufficient improvement" bar to replace
the served HGB: unseen-circuit MAE is tied and its intervals are under-covered (worse
calibration). The served model is **preserved**. This is logged as evidence, not a
model change.

## Recommended next step (promising, bounded)
Combine the quantile regressor's sharpness with proper calibration:
- **Conformalised quantile regression** (split-conformal on a held-out fold) to restore
  10–90 coverage at ~80% while keeping intervals far sharper than 23 laps; evaluate
  under the same LOCO protocol.
- Add **circuit-descriptive features** (pit-loss time, lap length, abrasiveness,
  typical stop count) from a static reference table — these *can* help unseen circuits,
  unlike `EventName`. This needs a curated data source, not FastF1 timing.
Only promote if a calibrated candidate beats the HGB on unseen-circuit MAE **and**
coverage-at-width.
