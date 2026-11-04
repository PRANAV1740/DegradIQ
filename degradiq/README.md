# DegradIQ 2026

DegradIQ keeps its six-stage tyre-degradation pipeline and adds an experimental first-pit-stop forecasting service. The feature estimates the per-lap probability of a driver's **first pit stop** from historical race laps. It is a probabilistic forecast, not an optimal-strategy guarantee.

## Local development

From the repository root:

```bash
# Core API, model and data dependencies
pip install -r degradiq/requirements.txt

# Optional dashboard UI
pip install -r degradiq/requirements-dashboard.txt

# Unified dashboard (both pages: First-Stop Prediction + Tyre Degradation)
streamlit run degradiq/app.py

# Or run a single page standalone:
streamlit run degradiq/dashboard_first_stop.py   # offline, engine-driven
streamlit run degradiq/dashboard.py              # tyre degradation (needs FastF1)

# API during local development
uvicorn degradiq.api:app --reload
```

API docs are available at `http://127.0.0.1:8000/docs`.

## First-stop model workflow

Training is an offline/local operation because it downloads and processes historical race sessions. Do not train the model inside a Vercel request.

```bash
python -m degradiq.scripts.train_first_stop --start-year 2022 --end-year 2026 --holdout-year 2025
```

The script:

1. Loads completed race sessions from the requested years using FastF1.
2. Builds a per-lap hazard dataset: information available at the end of lap `k` predicts whether the driver's **first** pit stop occurs on lap `k+1`.
3. Performs a chronological season holdout (default: train before 2025, evaluate on 2025) and writes Brier score, log loss, ROC AUC where defined, and first-stop-lap MAE where possible.
4. Refits the deployable model on all successfully collected completed races.
5. Saves the model and metadata under `degradiq/artifacts/` and coverage/backtest reports under `degradiq/reports/`.

The training script may take a while and requires network access to FastF1. Missing sessions are logged, not fabricated. Review `degradiq/reports/first_stop_training_coverage.csv` and `first_stop_backtest.json` before deploying. A model artifact is **not included by default** because no live historical-data training was run while preparing this package. Until you train it, the prediction endpoint returns a clear `503 model not trained` response instead of fake predictions.

### Canonical validation experiment (first-stop v2) and reproduction

The authoritative first-stop validation experiment is **`run_v2_20261018`**, label
version `2.0-strategic-green-vsc-sc`. Its tracked, text-only provenance lives in
[`degradiq/reports/experiments/first_stop_v2/`](reports/experiments/first_stop_v2/CANONICAL.md)
(manifests, backtests, candidate metadata with model SHA-256, reconciliation audit,
and `provenance.json`); the binary artifacts stay in the git-ignored run directory
under `degradiq/cache/first_stop_v2_runs/`.

Reproduce it **offline, from cached data only (no network)**:

```bash
python -m degradiq.scripts.reproduce_first_stop_v2 --strict
```

This re-runs the committed `evaluate_candidate_chronological` logic once per
chronological holdout (2024 development, 2025 partial), writes an **isolated**
`run_v2_verify_<timestamp>/` directory (never overwriting the original run or
production artifacts), and compares regenerated metrics against the historical
outputs within a declared absolute tolerance of `1e-4`. The canonical candidates,
metrics, reconciliation counts, and evaluation limitations are documented in
[`CANONICAL.md`](reports/experiments/first_stop_v2/CANONICAL.md).

> These are **development / partial-validation** results, **not** blind-test
> performance: 2024/2025 were examined development periods, the 2025 holdout is
> partial (11/24 events), and circuit-level generalisation to unseen tracks is not
> established (`EventName` is a feature and circuits recur across seasons).
>
> The earlier `degradiq/artifacts/candidate_first_stop_model*` (trained 2026-10-15)
> and `degradiq/reports/candidate_first_stop_backtest*.json` describe a **superseded**
> candidate, retained for history only.

The app now serves the **promoted validated HGB candidate**
(`degradiq/artifacts/first_stop_model_v2_hgb.joblib`), produced with
`python -m degradiq.scripts.promote_candidate`. The engine prefers it only because it
is label-version `current` and schema-compatible; the production logistic-regression
model (`first_stop_model.joblib`) is left untouched and is used automatically if the
promoted file is removed. See the [model card](reports/experiments/first_stop_v2/MODEL_CARD.md).

### API endpoints

- `GET /health` — service health.
- `GET /calendar` — current 2026 calendar and pending events.
- `GET /strategy/first-stop/status` — whether a trained model artifact is present.
- `GET /strategy/first-stop/lineup` — the configured 22-driver full-season lineup and team mapping.
- `POST /strategy/first-stop/predict` — per-driver first-stop forecast.
- `GET /docs` — interactive OpenAPI interface.

Example request body for `POST /strategy/first-stop/predict`:

```json
{
  "event_name": "United States Grand Prix",
  "race_laps": 56,
  "mode": "pre-race",
  "drivers": [
    {"driver": "NOR", "team": "McLaren", "grid_position": 2, "starting_compound": "MEDIUM"},
    {"driver": "PIA", "team": "McLaren", "grid_position": 4, "starting_compound": "HARD"}
  ]
}
```

Only drivers in the configured 2026 full-season lineup are accepted. For live mode, send `mode: "live"` and provide `current_lap` for each driver; current tyre age/compound, position and recent pace-loss are optional but improve the information supplied to the model. Use data available at the time of prediction only.

Outputs include per-lap stop probabilities, cumulative probability of having stopped by each lap, probability of stopping before the finish, most likely lap, median/window conditional on stopping, and a ranking by expected stop lap. The ranking is a forecast ordering, not the probability that a driver will literally be the first car to pit.

## Vercel deployment

**Deploy the FastAPI service, not the Streamlit process.** Streamlit is intended for local research and interactive analysis; the Vercel Python runtime is a better fit for the API.

1. Push the project to GitHub after reviewing the generated model and reports.
2. In Vercel, import the repository and set **Root Directory** to `degradiq`.
3. Leave the framework/build settings on automatic detection. Vercel should discover `main.py` as the FastAPI entrypoint and install `requirements.txt`.
4. Ensure the trained file `degradiq/artifacts/first_stop_model.joblib` and its metadata are included in the repository/deployment. Do not deploy before training and reviewing the chronological holdout.
5. Deploy a **Preview** first. Test `/health`, `/calendar`, `/strategy/first-stop/status`, `/strategy/first-stop/lineup`, and `/docs`; send a small prediction request only once status says `ready`.
6. If the preview behaves correctly and the bundle remains within Vercel's Python function size limit, promote/deploy to Production.

The Vercel root-directory setting matters: it allows `main.py`, `api.py`, `requirements.txt`, and the model modules to remain inside `degradiq/`, respecting the project scope. No root-level `vercel.json`, `package.json`, or changes to the existing `src/`, `data/`, root requirements, or `outputs/` are required for the API-only deployment.

The current `/calendar` endpoint fetches the schedule at request time. The pit-stop prediction endpoint does **not** fetch historical races at request time; it loads the pre-trained artifact. For predictable serverless response times, train and package the model before deployment. The API may need a future cache/data-refresh workflow if you want automated updates after each Grand Prix.

## Existing six-stage tyre-degradation pipeline

1. **Ingest:** Read the current event schedule from FastF1. Load FP1, FP2 and FP3 and retain only full-season drivers, valid green-flag laps, and tyre age at least 4.
2. **De-fuel:** Add back `(LapNumber - 1) * 1.9 * 0.03` seconds.
3. **De-traffic:** Remove laps more than 1.5 seconds slower than the defuelled driver/stint median.
4. **De-evolve:** Fit a linear field pace trend against session time. Only a trend showing the field getting faster can be removed. The correction is capped at each stint's own observed range.
5. **Model:** Fit a deterministic piecewise-linear hinge curve with nonnegative wear slopes. Insufficient data and below-threshold results are explicit statuses.
6. **Recommend and validate:** Find where the fitted curve crosses 1.5 s and 2.5 s. Fit on practice only and evaluate on the race holdout after baseline-normalizing each stint. Because eligible laps require tyre age at least 4, tyre age 2 is often absent; validation then uses the earliest eligible age as the baseline.

## First-stop model limitations

- First-stop behavior is learned from historical race timing. It is not the same as tyre degradation and is not a causal or optimal-strategy model.
- The model is a discrete logistic-regression hazard with driver/team/circuit/compound and lap-state features. Unknown categorical values are handled, but predictions for unseen drivers/teams or changed regulations may be less reliable.
- Historical first-stop labels rely on `PitInTime`; races/sessions with missing or incomplete pit-in timing may be skipped or weakly represented.
- Pre-race mode uses starting grid and compound, plus historical patterns. Live mode can use current state. No future race information should be included in either mode.
- The chronological holdout is recorded before final refitting. Review holdout metrics and sample coverage; no performance guarantee is made.
- The ranking is by expected first-stop lap conditional on a stop occurring in the forecast horizon. It is not a joint probability that a driver will be the first to stop.
- Current tyre-degradation constraints remain unchanged: fuel burn 1.9 kg/lap, 0.03 s/kg, minimum tyre life 4, 1.5 s traffic cutoff, and 1.5/2.5 s pit-window thresholds.

## Scope

All new source files, training artifacts and generated reports belong inside `degradiq/`. The existing `src/`, `data/`, root `requirements.txt`, and `outputs/` remain untouched.
