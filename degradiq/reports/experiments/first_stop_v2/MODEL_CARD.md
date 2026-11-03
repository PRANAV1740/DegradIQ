# Model Card — DegradIQ first-stop hazard model (v2)

A per-lap discrete-hazard model that estimates the probability of a Formula 1
driver's **first genuine strategic tyre-changing pit stop** occurring on the next
lap, and derives a predicted stop lap with an uncertainty window.

## Model details
- **Task:** per-lap binary hazard (`StopNextLap`), converted to a survival curve
  `S(k)=∏(1−hazard)` and a predicted first-stop lap + 10–90 interval.
- **Canonical candidate architecture:** calibrated `HistGradientBoostingClassifier`
  (`CalibratedClassifierCV(sigmoid)`, calibrated with `GroupKFold(5)` by `EventName`
  on training data only). Hyperparameters and `model_sha256` are in
  [`provenance.json`](provenance.json) / `candidate_*_metadata.json`.
- **Currently deployed in the app:** the **production logistic-regression** model
  (`degradiq/artifacts/first_stop_model.joblib`), which predates label versioning
  (status `unversioned_legacy`). The validated HGB candidate is **not** promoted;
  promotion remains gated on a tracked artifact and this card.
- **Features (11):** `LapNumber, RaceLaps, LapFraction, TyreLife, GridPosition,
  Position, RecentPaceLoss_s, Compound, Driver, Team, EventName`. All are available
  at the end of the lap being scored (no future information).
- **Label version:** `2.0-strategic-green-vsc-sc` (see `../../pit_strategy.py`). First
  stop = first genuine strategic tyre change; drive/pass-throughs, confirmed red-flag
  entries, and lap ≤2 SC/red compulsory stops are excluded; unresolved timing is
  censored, not guessed.

## Intended use
- Pre-race and in-race **decision-support** for first-stop timing: per-lap stop risk,
  cumulative probability, a predicted lap and an uncertainty window.
- **Not** an optimal-strategy solver, a guarantee, or a live-timing product. Not for
  betting or any safety-critical use.

## Training data
- FastF1 historical race laps, 2022–2024 (+ partial 2025), certified to one label
  version. 79 events, 24,641 per-lap rows, 1,304 positives (~5% positive rate).
  Provenance and the exact certified dataset are described in `CANONICAL.md`.

## Evaluation (development / partial-validation — NOT a blind-season test)
From [`generalization_summary.json`](generalization_summary.json):

| Regime | MAE | Baseline MAE | AUC | Brier | 10–90 cov | width |
|---|---|---|---|---|---|---|
| Temporal 2024 (seen circuit) | 6.96 | 8.20 | 0.744 | 0.0449 | 0.84 | 21.3 |
| Temporal 2025 partial (seen) | 7.79 | 8.67 | 0.754 | 0.0464 | 0.84 | 22.9 |
| LOCO macro / 25 (unseen circuit) | 6.93 | — | 0.778 | 0.0540 | 0.85 | 23.3 |

- **Classification generalises** to unseen circuits (LOCO AUC ≈ seen-circuit holdout).
- **Timing is circuit-dependent:** on unseen circuits the stop-lap MAE matches/beats
  the historical-median baseline in only **5/25** circuits (strictly 4/25).
- **Intervals are conservative:** ~0.84 coverage at ~22-lap width (over-covered vs
  the 80% nominal), not sharp.

## Limitations & caveats
- 2024/2025 were examined during development; **no untouched blind season** has been
  evaluated. The 2025 holdout is partial (11/24 events; the rest are genuine FastF1
  cache misses).
- Probabilistic metrics use in-race per-lap features; stop-lap MAE/intervals are a
  pre-race forecast — the two information regimes are reported separately, never mixed.
- No weather, traffic/gap, or safety-car *features* beyond the eligibility gating; a
  documented open label question (FU-1, Austria) is unresolved by design.
- FastF1 data limits apply (e.g. deleted/track-limit laps are not detectable).
- The production LR and the validated HGB candidate differ; app forecasts currently
  come from the LR.

## Reproduce
```bash
python -m degradiq.scripts.reproduce_first_stop_v2 --strict      # metrics (bit-exact)
python -m degradiq.scripts.validate_generalization               # temporal + LOCO
```
Environment: Python 3.13, scikit-learn 1.9.1, numpy 2.5.4, pandas 2.3.3 (pinned in
`provenance.json`). See `CANONICAL.md` and `EVALUATION.md` for full detail.
