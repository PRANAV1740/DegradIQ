# DegradIQ

**An F1 tyre-degradation and pit-stop strategy intelligence platform.**

DegradIQ turns Formula 1 race data into strategy insight. It pairs a machine-learning model that forecasts a driver's first pit-stop window with an interactive strategy canvas, a tyre-degradation view, and a verified circuit map for every round of the 2026 calendar.

**Live app:** https://degrad-iq.vercel.app
&nbsp;&middot;&nbsp; **API docs:** https://degrad-iq.vercel.app/docs

> This is a research and portfolio project. It produces probabilistic forecasts, not optimal-strategy guarantees, and does not redistribute raw F1 timing data.

---

## What it does

- **First pit-stop prediction.** Estimates the per-lap probability of a driver's first pit stop and surfaces the most likely stop window.
- **Strategy Stint Canvas.** An interactive view of stints, compounds, and how the prediction evolves across a race.
- **Tyre Intelligence.** Degradation context derived from historical race laps.
- **Circuit maps.** A verified track map for all 23 rounds of the 2026 season, with corner numbers and marshal sectors where the source allows.

## The machine-learning approach

The first pit-stop is framed as a **discrete-time hazard problem**: for each lap, the model predicts how likely the stop is to happen on that lap, rather than guessing the exact lap directly. Those per-lap probabilities are then turned into a predicted stop window.

- **Model:** a calibrated `HistGradientBoostingClassifier` (scikit-learn), chosen for the non-linear, tabular nature of strategy data such as tyre age, compound, pace, and track conditions.
- **Validation:** leave-one-circuit-out (GroupKFold on event name), so the model is always evaluated on tracks it has never trained on.
- **Baseline:** a Logistic Regression model is kept alongside for interpretability.
- **Honesty:** missing sessions are logged, never fabricated. Until a model is trained, the prediction endpoint returns a clear "model not trained" response instead of fake predictions.

## Circuit geometry pipeline

Accurate track maps were one of the hardest parts of the build. DegradIQ uses a multi-source pipeline with validation at every step, so each map is accurate and properly attributed:

1. **MultiViewer API** for verified outlines with real corner numbers and marshal sectors.
2. **OpenStreetMap** (via Overpass, node-topology validated) for circuits without a MultiViewer source.
3. **MIT-licensed GeoJSON** for brand-new venues such as the 2026 Madrid circuit.

Every circuit is classified as **VERIFIED**, **PARTIAL**, **UNAVAILABLE**, or **FAILED**. The system never invents geometry: each map records its own source and licence, and anything that cannot be verified is labelled honestly instead of faked. See [`degradiq/web/data/DATA_SOURCES.md`](degradiq/web/data/DATA_SOURCES.md) for full provenance and licensing.

## Architecture

```
FastAPI service (degradiq/api.py)
  ├─ Prediction engine ....... calibrated HGB hazard model + feature pipeline
  ├─ Circuit intelligence .... multi-source geometry build (MultiViewer / OSM / GeoJSON)
  ├─ Tyre degradation ........ six-stage historical-lap pipeline
  └─ Web UI (degradiq/web) ... vanilla JavaScript frontend, same-origin

Streamlit dashboards (degradiq/app.py) ... local research UI
Deployment ............................... Vercel (continuous), GitHub Actions CI
```

## Quick start

Requires Python 3.12+.

```bash
# Install core API, model, and data dependencies
pip install -r degradiq/requirements.txt

# Run the API locally
uvicorn degradiq.api:app --reload
# API docs at http://127.0.0.1:8000/docs

# Optional: local research dashboards
pip install -r degradiq/requirements-dashboard.txt
streamlit run degradiq/app.py
```

Training the first-stop model is an offline operation that downloads historical sessions with FastF1. See [`degradiq/README.md`](degradiq/README.md) for the full training and reproduction workflow.

## Key API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Service health check |
| GET  | `/calendar` | 2026 season calendar |
| GET  | `/strategy/first-stop/status` | Whether a trained model is available |
| GET  | `/strategy/first-stop/lineup` | Driver lineup for prediction |
| POST | `/strategy/first-stop/predict` | First pit-stop probability forecast |
| GET  | `/api/intel/index` | Circuit coverage index for all 2026 rounds |
| GET  | `/api/intel/circuit/{circuit_id}` | Geometry and metadata for one circuit |
| GET  | `/api/intel/tyre/{event_id}` | Tyre degradation data for one event |

## Testing

```bash
python -m pytest degradiq/tests/ -q
```

The suite covers the prediction engine, circuit-geometry integrity gates, provenance, and the public API. Dependencies are version-pinned (including `fastf1==3.9.2`) for reproducible builds, and the suite runs in GitHub Actions on every push.

## Tech stack

Python, scikit-learn, FastF1, Pandas, NumPy, FastAPI, Streamlit, JavaScript, Vercel, GitHub Actions.

## Data sources and licensing

DegradIQ derives from several sources, each with its own terms: MultiViewer (non-commercial fan app), OpenStreetMap (ODbL), bacinger/f1-circuits (MIT), and FastF1-cached F1 timing and telemetry. Only normalised outlines and derived features are packaged, never raw F1 data. Full details are in [`degradiq/web/data/DATA_SOURCES.md`](degradiq/web/data/DATA_SOURCES.md). Before any commercial use, redistribution rights for the F1-derived sources should be confirmed in writing.
