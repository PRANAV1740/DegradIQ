# DegradIQ — deployment readiness

Status of the checks required before deploying. **No public deployment has been
performed** (that needs a Vercel account/authorisation and is the owner's call); this
documents that the project is ready and how to deploy it.

## What runs where
- **API (deploy target):** FastAPI, entry `degradiq/main.py` → `degradiq/api.py`. Serves
  the validated model through the shared `PredictionEngine`. **No FastF1 at runtime**
  (verified by the serverless-isolation test), so the function stays lean.
- **Dashboard (local/research):** Streamlit, `degradiq/app.py` (multipage). Intended for
  local use, not the serverless deploy.

## Verified locally (this environment)
| Check | Result |
|---|---|
| Dependency spec present | ✅ `degradiq/requirements.txt` (API), `requirements-dashboard.txt` (UI), `requirements-train.txt` (offline training) |
| Backend imports + starts | ✅ `from degradiq.api import app`; `/health` → 200 |
| End-to-end smoke | ✅ `POST /strategy/first-stop/predict` → 200, served model `calibrated_hgb`, returns probabilities + `interval_width_if_stops` |
| Dashboard starts | ✅ `AppTest` renders `degradiq/app.py` and both pages without exceptions |
| Shared engine | ✅ API and dashboard both use `PredictionEngine`/`load_default_engine` |
| Secrets | ✅ none in the repo; **no API keys required**. FastF1 needs internet only for the *local* tyre-degradation page and offline data prep |
| Artifact bundle size | ✅ ~4 MB total in `artifacts/` (production LR + validated HGB + metadata) — well within serverless limits |
| Function leanness | ✅ `.vercelignore` excludes `cache/`, `reports/`, `tests/`, `scripts/`, the Streamlit dashboard and training deps; **keeps** `artifacts/` |

## Run locally
```bash
pip install -r degradiq/requirements-dashboard.txt   # everything (API + UI + data)
uvicorn degradiq.api:app --reload                     # backend → http://127.0.0.1:8000/docs
streamlit run degradiq/app.py                         # dashboard (First-Stop + Tyre Degradation)
```
API-only (lean): `pip install -r degradiq/requirements.txt`.

## Deploy the API to Vercel (owner action)
1. Push to GitHub; review the generated model and reports.
2. In Vercel, import the repo, set **Root Directory** to `degradiq`.
3. Automatic detection finds `main.py` (FastAPI) and installs `requirements.txt`.
4. Ensure `artifacts/first_stop_model_v2_hgb.joblib` (+ metadata) and the production
   `first_stop_model.joblib` ship (they are not ignored).
5. Deploy a **Preview**; test `/health`, `/calendar`, `/strategy/first-stop/status`
   (should report `calibrated_hgb`, label `current`), `/lineup`, `/docs`, and one small
   `/predict`. Promote to Production only if the preview is correct.

## Environment & secrets
- No environment variables are required to serve predictions.
- `VERCEL=1` (set by Vercel) routes any FastF1 cache to `/tmp` — but the API does not
  fetch FastF1 at request time, so this only matters if live ingestion is added later.
- No secrets are stored or needed; do not add any to the repo.

## Not verified / out of scope here
- A real Vercel preview/production deployment (needs account access).
- Live timing (blocked: FastF1's `livetiming.formula1.com` returns 403 here — the app
  exposes an honest live-unavailable state and uses historical replay instead).
- Load/performance testing.
