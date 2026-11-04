# 2025 blind-season test — attempt and blocker

**Outcome: not completed. No genuine blind test was performed.** The attempt is
recorded here honestly; the 11 available 2025 events are the *previously examined*
set, not a blind test.

## What was attempted
`python -m degradiq.scripts.blind_2025_eval` recovers the 13 previously unavailable
2025 events into an **isolated** checkpoint directory (never the legacy dir; the
original run and the production model are untouched), then evaluates the **frozen**
2022–2024 model (identical to the promoted served model) on them — the 13 events
never influenced training or model selection, so they would be a true out-of-sample
test.

## What happened
| | count |
|---|---|
| Events successfully recovered | **0 of 13** |
| Events still unavailable | **13** (all classified `cache_miss`) |
| Events evaluated | 11 (already cached — the *previously examined* set) |

The 13 unavailable events: British, Belgian, Hungarian, Dutch, Italian, Azerbaijan,
Singapore, United States, Mexico City, São Paulo, Las Vegas, Qatar, Abu Dhabi.

## Why (data-source blocker — verified)
FastF1 serves historical **lap-timing** data from `livetiming.formula1.com`. From this
environment that host returns **HTTP 403 Forbidden**, so no un-cached race can be
downloaded. The schedule/results API (jolpica/ergast) is reachable, which is why event
*names* resolve but the timing *payloads* do not. The committed cache-miss discriminator
correctly classified all 13 as `cache_miss` (only a ~0.5 KB `session_info.ff1pkl` is
present locally, no timing file).

This is an environment/access limitation, not a code defect and not a model result.

## The 11 evaluated events are NOT a blind test
The 11 recoverable events are exactly the ones examined during v2 development. On them
the frozen model reproduces the earlier 2025-partial figures (MAE 7.79, AUC 0.754,
Brier 0.0464, 10–90 coverage 0.843, width ~23) — i.e. nothing new is learned, and
these must **not** be reported as blind-season performance. Full per-field numbers are
in [`blind_2025_summary.json`](blind_2025_summary.json) (`recovered_events: []`,
`blind_13_recovered: no rows`).

## How to complete it (network-enabled environment)
Run the same command where `livetiming.formula1.com` is reachable:

```bash
python -m degradiq.scripts.blind_2025_eval
```

It will download the 13 events (rate-limited), evaluate the frozen model, and write
`blind_13_recovered` metrics (AUC/Brier/log-loss, per-event MAE vs the historical-median
baseline, interval coverage and width) alongside `examined_11` and `full_2025`. Only
then is a genuine partial-blind 2025 result available.

## Status
The 2025 blind-season test remains **pending on data availability**. The genuinely
untouched alternative is a locked **2026** prospective holdout (new driver lineup),
pre-registered before inspection — see the execution log.
