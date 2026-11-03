# First-stop v2 validation — canonical experiment

This directory is the **tracked, authoritative provenance** for the first-stop
v2 validation experiment (`run_v2_20261018`). The binary artifacts (per-event
hazard checkpoints, `combined_certified.parquet`, candidate `*.joblib`) stay in
the git-ignored run directory under `degradiq/cache/first_stop_v2_runs/`; only
the small text manifests, reports and metadata are tracked here.

## Canonical candidates

| Tag | Train seasons | Holdout | Model SHA-256 |
|---|---|---|---|
| `2024_dev` | 2022–2023 | 2024 (24/24 events) | `51ae3bae…e0a5e` |
| `2025_partial` | 2022–2024 | 2025 (11/24 events, **partial**) | `724ab2eb…3be38` |

Label definition: `2.0-strategic-green-vsc-sc` · diagnostics schema: `1.0`.
Feature schema and library versions are recorded in [provenance.json](provenance.json).

**Superseded:** `degradiq/artifacts/candidate_first_stop_model*` (trained
2026-10-15) and `degradiq/reports/candidate_first_stop_backtest*.json` describe an
**earlier** candidate and are **not** the canonical v2 result. They are retained
for history only. The production model `degradiq/artifacts/first_stop_model.joblib`
is a separate, unevaluated logistic-regression artifact and was not touched.

## Reproduce (offline, cached data only — no network)

```bash
python -m degradiq.scripts.reproduce_first_stop_v2 --strict
```

Required inputs (already present as cached data; nothing is downloaded):

- `degradiq/cache/first_stop_v2_runs/run_v2_20261018/combined_certified.parquet`
- `degradiq/cache/first_stop_v2_runs/run_v2_20261018/coverage.csv`
- `degradiq/cache/first_stop_v2_runs/run_v2_20261018/reconciliation_audit.csv`

The script re-runs the committed `evaluate_candidate_chronological` logic once per
holdout, writes an **isolated** `run_v2_verify_<timestamp>/` directory (never
overwriting the original run or production artifacts), and writes
`reproduction_report.json` comparing regenerated metrics against the historical
outputs within a declared absolute tolerance of `1e-4`.

### Last verified reproduction

Both candidates reproduced **bit-exactly** (every metric `|diff| = 0.0`, and the
regenerated model SHA-256 is byte-identical to the historical hashes) at git
revision `c9d6db1` with the pinned library versions in `provenance.json`.

| Metric | 2024_dev | 2025_partial |
|---|---|---|
| First-stop lap MAE | 6.9631 | 7.7921 |
| Baseline (historical median) MAE | 8.2015 | 8.6657 |
| Brier | 0.0449 | 0.0464 |
| Log loss | 0.1782 | 0.1810 |
| ROC-AUC | 0.7436 | 0.7544 |
| ECE | 0.0184 | 0.0075 |
| 10–90 interval coverage | 0.8403 | 0.8427 |

## Evaluation limitations (do not overstate)

These are **development / partial-validation** results, **not** blind-test
performance:

- 2024 and 2025 were examined development/validation periods, not an untouched
  blind season.
- The 2025 holdout is **partial** (11 of 24 events); it is not a full-season result.
- Probabilistic metrics (Brier/log-loss/AUC) are measured over per-lap rows using
  in-race features, while stop-lap MAE is a **pre-race** forecast — two different
  information regimes.
- 10–90 interval coverage is reported above its 80% nominal target, i.e. the
  intervals are conservative (wide), not exactly calibrated.
- `EventName` is a model feature and the same circuits recur across seasons, so
  circuit-level generalisation to unseen tracks is **not** established here.

## Reconciliation counts (one dataset, three lenses)

The three historically reported "identical/different" splits are three comparison
definitions over the single preserved [reconciliation_audit.csv](reconciliation_audit.csv)
(79 events), not conflicting evidence:

| Definition | Identical | Different |
|---|---|---|
| Feature **+ target** equal (ignores row membership) | 55 | 24 |
| Row-membership equal (matches `manifest.json`) | 31 | 48 |
| Rows **and** features **and** target equal (`auth_status`) | 28 | 51 |
