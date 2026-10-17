# DegradIQ — Audit Follow-ups (open investigations)

Status: open. Recorded during the Phase 1 correction pass. No label, filtering, or
dataset changes have been made for the items below — they require a modelling-objective
decision first.

## FU-1 — Austria 2023 first-stop target definition (UNRESOLVED)

### The question
Does the first-stop hazard target **deliberately exclude** first stops that occur under
VSC / Safety Car (or on otherwise ineligible laps), or is it **unintentionally missing
legitimate pit-stop events**? The intended modelling objective must be settled before any
label or filter change is proposed.

### What the local evidence proves
Source: `degradiq/cache/hazard_checkpoints/session_2023_Austrian_Grand_Prix.parquet`
(248 eligible rows, 20 drivers) and `degradiq/pit_strategy.py::build_hazard_rows`.

- Only **7 of 20 drivers** have a labelled first stop (`StopNextLap == 1`). The 7 labels
  are internally consistent genuine stops (e.g. VER lap 23 → MEDIUM, PER lap 24 → MEDIUM;
  tyre-age reset and/or compound change present).
- **15 of 20 drivers' eligible rows span only laps 4–12**, then terminate with no label.
  The label for a driver's first stop attaches to the *lap before* the stop, and a row is
  created only when that lap is green and accurate:
  - `TrackStatus == "1"` required (`pit_strategy.py`, green-only row filter), and
  - `IsAccurate` must be true.
  So if the pre-stop lap is under yellow/VSC/SC or flagged inaccurate, no positive row is
  generated for that driver — this is the mechanism that suppresses the count.
- The exclusion rules added in commit `2313d00` only *remove* pit visits (lap ≤ 2 under
  SC/red flag; SC/red-flag visits with no compound change and no tyre reset). They cannot
  *inflate* counts and are not the cause of the low number.

### What the local evidence does NOT prove
- It does **not** establish the *complete* cause of the 7-positive count. The exact
  per-lap `TrackStatus` sequence for Austria 2023 was **not** reconstructed from the
  FastF1 HTTP cache in this pass (out of scope), so the precise set of laps filtered out
  cannot be enumerated from what is recorded here.
- The previous audit's claim that "a Safety Car routed the entire field through the pit
  lane" is **not** supported by the code or the parquet, and is inconsistent with the
  official 2023 Austrian GP chronology (a largely green race, decided post-race on
  track-limits penalties). Treat that explanation as an unverified inference, not fact.

### Distinction to preserve
- **Official event chronology** (what actually happened in the race) is a separate source
  of truth from **local evidence** (parquet rows + `TrackStatus` / `IsAccurate` / compound
  changes / tyre-age resets). The local evidence shows *lap-eligibility truncation*, which
  is necessary but not by itself sufficient to conclude the race's true first-stop pattern.

### Decision needed before any change
1. Define the intended target: "first *green-flag, accurate-lap* stop" vs "first *actual*
   pit stop regardless of track status". The current filter implements the former.
2. If the latter is intended, decide how VSC/SC/inaccurate pre-stop laps should contribute
   a label without reintroducing compulsory red-flag tyre changes as strategic stops.
3. Only after (1)–(2): propose label/filter changes, re-derive affected checkpoints, and
   re-evaluate. Do not change labels in the current phase.

### Suggested verification step (future, read-only)
Load the raw Austria 2023 race laps from the FastF1 cache and tabulate, per driver, the
first `PitInTime` lap alongside that lap's `TrackStatus` and `IsAccurate`, to enumerate
exactly which legitimate stops are dropped by the eligibility filter.
