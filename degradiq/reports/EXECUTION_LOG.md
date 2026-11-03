# DegradIQ — Execution Log

Rolling record of autonomous execution: completed work, decisions, tests, commits,
and the next executable task. Newest milestone last.

## Baseline (verified)

- Repo `PRANAV1740/DegradIQ`. P0 merged: `main` at `246bba5` (merge of PR #4),
  underlying `dd4f1a2`. Label `2.0-strategic-green-vsc-sc`. Canonical v2 run
  `run_v2_20261018` reproduces bit-exactly (see `experiments/first_stop_v2/`).
- Full suite green at start of this phase.

## Milestone: Phase I — synchronisation + baseline (done)

- Verified PR #4 MERGED; synced; created feature branch `feat/research-validation`
  from `origin/main`.
- Inspected app layer: FastAPI (`degradiq/api.py`), Streamlit (`degradiq/dashboard.py`),
  config, pipeline, ingest. **Decision/finding:** the API currently serves the
  **unevaluated production logistic-regression** model (`first_stop_model.joblib`),
  not the validated calibrated-HGB candidate — a Phase IV integration gap, recorded
  below.

## Milestone: Phase II — independent generalisation validation (done)

New offline, deterministic harness `degradiq/scripts/validate_generalization.py`
(reuses committed training/eval; no network, no re-labelling, no methodology change):

- **Temporal holdout** (unseen season, seen circuit) reproduces exactly
  (MAE 6.96/7.79, AUC 0.744/0.754) and now also reports **10–90 interval width**
  (~21–23 laps) — the audit's missing metric.
- **Leave-one-circuit-out (LOCO)** over 25 circuits (genuine unseen-circuit test).
- Results (see `experiments/first_stop_v2/EVALUATION.md`,
  `generalization_summary.json`):
  - Classification **generalises** to unseen circuits (LOCO AUC 0.742 micro / 0.778
    macro; Brier 0.048–0.054) ≈ seen-circuit holdout → the `EventName`-recurrence
    optimism concern is largely not borne out.
  - Stop-lap **timing MAE** beats the historical-median baseline in only **5/25**
    unseen circuits → the timing edge is partly circuit memorisation (worst: Monaco,
    Canada, Azerbaijan, Australia).
  - Intervals are **conservative** (0.84 coverage at ~22-lap width), not sharp.
- **FU-1 (Austria 2023):** offline probe confirms earliest first-pit entries are
  lap ≤2 under SC (correctly excluded); mechanism is eligibility filtering, not a
  bug. Underlying objective choice left **open** (not silently changed).
- **2025 data completeness:** the 13 missing events are **genuine cache misses**
  (verified). Recovery needs a network download and must go to an **isolated** dir
  (`recover_2025.py` writes into the legacy dir — do not use as-is). Deferred with a
  documented plan; a complete 2025 set is still not a clean blind test.
- Tests: `degradiq/tests/test_generalization.py` + existing provenance/label tests.

## Protected-artifact ledger (unchanged)

`artifacts/` (production + superseded candidate), `cache/hazard_checkpoints/`
(94 legacy), and `run_v2_20261018/` were **not** modified. New experiment outputs go
to git-ignored `cache/first_stop_v2_runs/validation_*/`.

## Continuation checkpoint — next executable tasks (Phases III–VIII)

Branch `feat/research-validation`; last commit recorded in git. Resume here.

- **P1 · Isolated 2025 recovery + locked blind test.** Add a recovery entry point
  that writes to a NEW checkpoint dir (never the legacy dir), uses the committed
  rate limiter + `build_hazard_rows`, re-builds a certified dataset, and re-runs the
  2025 holdout. Pre-register the protocol. For a *genuine* blind test, lock a 2026
  prospective window (new driver lineup in `config.py`) and evaluate once.
- **P2 · Phase III model comparison.** Compare calibrated-HGB vs calibrated-LR vs a
  discrete-time survival baseline under identical splits (incl. LOCO). Select on
  generalisation + calibration, not dev score. Keep a candidate registry.
- **P2 · Phase IV prediction engine.** Wrap the validated candidate behind a stable
  `PredictionEngine` (load once, compatibility-check feature schema + label version,
  per-lap hazard + cumulative + pre-race timing + intervals-with-width + provenance).
  Point the API/dashboard at it instead of the unevaluated LR.
- **P2 · Phase V dashboard.** Extend the Streamlit app: overview, tyre degradation,
  first-stop prediction (prob + cumulative + lap + interval width), race context,
  historical replay (point-in-time only), evaluation views. No fake predictions;
  clear mode/freshness indicators.
- **P3 · Phase VI live adapter**, **Phase VII CI** (run the suite on push),
  **Phase VIII docs/model card**.

**Single most valuable next step:** Phase IV prediction-engine interface serving the
*validated* candidate — it unblocks the whole application track and removes the
current unevaluated-LR gap.
