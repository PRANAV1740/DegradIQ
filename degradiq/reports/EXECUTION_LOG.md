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

## Milestone: PR #5 self-review + merge (done)

Independent re-review of PR #5 found one real defect: the per-circuit LOCO figures
and the "5/25" timing claim cited in `EVALUATION.md` lived only in a git-ignored
full report. Fixed by promoting `leave_one_circuit_out.per_circuit` and a
`timing_mae_vs_baseline` block (5 ≤, 4 strict, of 25) into the tracked
`generalization_summary.json`, with tests. Science re-verified: LOCO groups are 25
distinct physical circuits (EventName↔circuit 1:1); held-out circuit excluded from
train + calibration; temporal split leakage-safe and reproduces 2024 exactly. The
dashboard-test timeout was **not** reproduced in a clean full run (82 passed) →
environmental/contention, non-regressive; test left unchanged. Merged to `main`:
`fae1d3a` (PR #5), underlying `03ccb41`, `16f4777`.

## Milestone: Phase IV — PredictionEngine (started)

- Added `degradiq/prediction_engine.py`: load-once, compatibility-checked interface
  (`PredictionEngine` + `load_default_engine`). Validates feature schema == current
  `FEATURE_COLUMNS` (else `IncompatibleModelError`); classifies label version
  (`current` / `unversioned_legacy` / `stale`); exposes `provenance()` (path, sha256,
  schema, label status, seasons, architecture) and `predict(...)` which reuses the
  committed `predict_first_stops` and adds the 10–90 interval **width** per driver
  plus an `engine` block distinguishing `feature_mode` (pre-race/live) from
  `context_mode` (historical/replay/live).
- Integrated `degradiq/api.py` onto the engine (lazy, cached, fail-soft 503). The
  **production LR model is unchanged and still the default** — no candidate promoted.
  Response keys are additive (backward compatible); serverless-isolation test passes.
- Tests: `degradiq/tests/test_prediction_engine.py` (11) + end-to-end API smoke
  (status + predict 200 with provenance and interval width). Full suite: 93 passed.
- **Not yet done:** Streamlit dashboard integration; replay point-in-time helper;
  loading a *validated candidate* (promotion still gated on a tracked artifact +
  decision). Production LR remains protected.

## Milestone: Phase V — first-stop dashboard (started)

- Added `degradiq/dashboard_first_stop.py`: an offline, engine-driven Streamlit page
  (`streamlit run degradiq/dashboard_first_stop.py`). Sections: model status +
  provenance (label-version badge, sha256, seasons); first-stop prediction
  (per-driver P(stop), most-likely/median lap, 10–90 window and **width**, plus a
  cumulative-probability chart); and model evaluation read from the tracked
  `generalization_summary.json` (temporal vs LOCO, with the honest 5/25-timing and
  wide-interval caveats). No fake predictions — a missing artifact is stated.
- Deliberately a **separate** page from the existing network-heavy tyre-degradation
  dashboard (`dashboard.py`, unchanged) so it stays offline and testable.
- Tests: `degradiq/tests/test_dashboard_first_stop.py` (AppTest). Full suite: 95 passed.
- **Not yet done:** merge the two dashboards into one multipage app; tyre-degradation
  + race-context + point-in-time historical replay sections for the engine page;
  live adapter (Phase VI); CI (Phase VII); model card (Phase VIII).

## Milestone: replay + model card + CI (done)

- **Historical replay (point-in-time):** added `PredictionEngine.score_rows()` (scores
  actual per-lap rows) and a replay section in `dashboard_first_stop.py` driven by a
  tracked offline fixture `experiments/first_stop_v2/replay_sample_spanish_2024.csv`
  (Spanish GP 2024, 266 rows, 20 drivers). A lap slider reveals per-lap hazard +
  cumulative stop probability using only laps ≤ the replay lap, with the actual first
  stop marked once reached. Fully offline and tested.
- **Model card (Phase VIII):** `experiments/first_stop_v2/MODEL_CARD.md` — details,
  intended use, data, metrics (temporal + LOCO), and honest limitations, all backed by
  tracked JSON.
- **CI (Phase VII):** `.github/workflows/tests.yml` runs the full suite on push/PR
  (Python 3.13, dashboard requirements). The suite is self-contained (synthetic
  fixtures + tracked artifacts); no network needed.
- Tests: `test_prediction_engine.py` extended (score_rows on the fixture + missing-cols
  guard). Full suite: 97 passed.

## Milestone: Phase VI — live/replay data-source adapter (done)

- Added `degradiq/live_adapter.py`: a `LiveSource` boundary separating *where* per-lap
  feature rows come from from *how* they are scored. `UnavailableLiveSource` is the
  honest default (no real-time feed — FastF1 is historical, and the official
  live-timing endpoint needs per-session auth, 401 unauthenticated); it returns no
  rows and says why. `ReplayLiveSource` drives point-in-time replay from a recorded
  frame (never exposes future laps) with status/freshness. `refresh_prediction()`
  scores only what a source can currently provide — no fabricated predictions.
- Dashboard shows an honest **live-data-unavailable** banner.
- Tests: `degradiq/tests/test_live_adapter.py` (7). Full suite: 104 passed.
- CI (added in the previous PR) is green on `main`.

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

**Single most valuable next step:** wire the Streamlit dashboard onto the
`PredictionEngine` (overview, tyre degradation, first-stop prediction with interval
width, race context, point-in-time historical replay, evaluation views), then gate a
validated-candidate promotion (tracked artifact + model card) before swapping the
production default. The engine interface (Phase IV core) now exists to build on.
