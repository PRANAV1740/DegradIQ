# First-stop v2 — independent generalisation evaluation

Reproducible offline evaluation of the canonical v2 candidate, reusing the committed
training/evaluation logic on the certified, label-versioned dataset
(`run_v2_20261018/combined_certified.parquet`, 24,641 rows / 1,304 positives,
label `2.0-strategic-green-vsc-sc`). No network, no re-labelling, no change to model
methodology.

```bash
python -m degradiq.scripts.validate_generalization            # temporal + LOCO
python -m degradiq.scripts.validate_generalization --temporal-only
```

Outputs: full per-circuit report in a git-ignored
`degradiq/cache/first_stop_v2_runs/validation_<ts>/`, tracked headline in
[`generalization_summary.json`](generalization_summary.json).

## Why two regimes

`EventName` is a model feature and the same circuits recur across seasons, so the
chronological (temporal) holdout only measures **unseen season, seen circuit**. To
test genuine **unseen-circuit** generalisation we add **leave-one-circuit-out
(LOCO)**: for each circuit C, train on every other circuit (all seasons) and test
on C; C never appears in training.

| Regime | Answers | Caveat |
|---|---|---|
| Temporal holdout (2024, 2025) | Unseen *season*, circuit already seen | 2025 holdout is partial (11/24) and was examined in development |
| Leave-one-circuit-out | Unseen *circuit* | Mixes seasons within a circuit (spatial, not temporal, question) |

Information regimes are kept separate: probabilistic metrics (Brier/log-loss/AUC)
use in-race per-lap features; stop-lap MAE and 10–90 interval metrics are a
**pre-race** forecast from starting grid/compound/tyre-age only. They are never mixed.

## Headline results

| Regime | MAE | Hist-median MAE | AUC | Brier | 10–90 cov | 10–90 width (laps) |
|---|---|---|---|---|---|---|
| Temporal 2024 (seen circuit) | 6.96 | 8.20 | 0.744 | 0.0449 | 0.840 | 21.3 |
| Temporal 2025 partial (seen circuit) | 7.79 | 8.67 | 0.754 | 0.0464 | 0.843 | 22.9 |
| **LOCO micro-pooled (unseen circuit)** | 6.97 | — | 0.742 | 0.0481 | — | — |
| **LOCO macro over 25 circuits (unseen)** | 6.93 | — | 0.778 | 0.0540 | 0.847 | 23.3 |

## What this establishes (and what it does not)

1. **Classification generalises to unseen circuits.** LOCO AUC (0.742 micro / 0.778
   macro) and Brier (0.048–0.054) are close to the seen-circuit temporal holdout
   (AUC 0.744–0.754, Brier 0.045–0.046). The audit's concern that `EventName`
   recurrence makes the discrimination metrics optimistic is **largely not borne
   out** — the model still ranks per-lap stop risk on circuits it has never seen.

2. **The stop-lap TIMING advantage is partly circuit memorisation.** On unseen
   circuits the pre-race stop-lap MAE beats the historical-median baseline in only
   **5 of 25 circuits**. The clear MAE edge in the temporal holdout (6.96 vs 8.20)
   therefore depends substantially on having seen the circuit before. Best unseen
   circuits: Bahrain (3.8), French (4.1), Spanish (4.2). Worst: Monaco (11.6),
   Canada (11.2), Azerbaijan (9.0), Australia (8.7) — street / variable-strategy
   tracks, as expected.

3. **Intervals are conservative, not sharp.** 10–90 coverage of ~0.84–0.85 is
   achieved with ~21–23-lap-wide intervals in every regime. High coverage reflects
   wide intervals (over-coverage vs the 80% nominal), not precise timing. Interval
   **width** is now reported alongside coverage so this cannot be read as calibrated
   sharpness.

**Net:** the model is a defensible per-lap *risk* model that generalises across
circuits; its *timing* (stop-lap) usefulness is strongest on familiar circuits and
degrades to around baseline on unfamiliar ones. Claims should be scoped accordingly.

## Data completeness (2025) — status

The 13 "unavailable" 2025 events in the v2 run are **genuine cache misses**, not
processing failures: their FastF1 Race directories contain only a ~0.5 KB
`session_info.ff1pkl` with no timing payload (verified with the committed
`_race_timing_cached` discriminator). Recovering them requires a network download
(outbound network is reachable in principle; `livetiming.formula1.com` live
endpoints are not, which is expected). This was **not** run in this pass because:

- it writes new checkpoints and must go to an **isolated** run directory (the
  committed `recover_2025.py` writes into the legacy checkpoint dir and would
  violate artifact isolation — do not run it as-is for this purpose); and
- even a complete 2025 set is **not a clean blind test**, because the 2025_partial
  candidate was trained and examined during development.

**Recommended genuine blind test:** a *locked prospective* holdout on a season/period
that has not influenced model development (e.g. 2026, in progress), with the
evaluation protocol pre-registered before the locked results are inspected. See the
continuation checkpoint for the isolated-recovery and locked-holdout plan.

## FU-1 (Austria 2023 label count) — still open

An offline probe (cache present) tabulated each driver's first `PitInTime` lap and
its confirmed track-status state. The earliest first-pit entries cluster at lap ≤2
under Safety Car (`entry_state=4`) — exactly the non-strategic early stops the label
logic deliberately excludes (lap ≤2 SC rule, and the no-tyre-change rule). This
confirms the FU-1 mechanism is **eligibility filtering**, not a code defect: the
naive "first pit" is not the first *strategic* stop. Whether genuine later green/VSC
strategic stops are under-captured remains a **modelling-objective decision**
(documented in `../../audit_followups.md`), not a bug, and is intentionally left open
rather than silently changed.
