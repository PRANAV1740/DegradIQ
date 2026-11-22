# Circuit Intelligence — data sources & licensing

Every circuit map in DegradIQ is built from a trustworthy external source with recorded
provenance. Geometry is never fabricated; a circuit with no usable source is shown as
*unavailable*. Each packaged circuit JSON records its own `source_dataset`, `licence`,
`identity_validation` and retrieval metadata.

This is a non-commercial research / portfolio project. The notes below are a good-faith
licensing summary, not legal advice; before any commercial use, redistribution rights for the
F1-derived sources should be confirmed in writing.

## Providers (in priority order)

| # | Source | What it provides | Licence / terms |
|---|---|---|---|
| 1 | **MultiViewer circuit API** (`api.multiviewer.app`) | Outline + numbered corners + marshal sectors + rotation. Resolved per event by the OpenF1 `circuit_key`. This is the same circuit-info source FastF1 exposes. | MultiViewer is a fan-made, **non-commercial** app; underlying F1 timing/telemetry is subject to F1's rights. Only a normalised outline + annotations are packaged — not raw telemetry. |
| 2 | **OpenStreetMap** via the Overpass API | `highway=raceway` circuit outlines (node-topology validated). Used for circuits MultiViewer lacks (e.g. the 2026 Sepang venue). | © OpenStreetMap contributors, **ODbL 1.0** (share-alike). Full way/relation provenance recorded; see `OSM_ATTRIBUTION.md`. |
| 3 | **bacinger/f1-circuits** GeoJSON | Closed circuit outlines for brand-new venues with no other source (e.g. the 2026 Madrid *Madring*). | **MIT** licence (attribution required). Traced from Google My Maps / Wikipedia — an approximate outline, not surveyed. |
| — | **FastF1** cached telemetry | Fastest-lap racing-line + corners for circuits whose session telemetry is cached locally. | F1 timing/telemetry; only derived outline + corner numbers packaged. |

## Identity & integrity

- Every circuit's geometry is matched to its configured identity (circuit name, distinctive
  corner names, or a reviewed source id) — geographic proximity alone is never accepted.
- A known calendar correction is encoded: the **2026 "Bahrain Grand Prix" is held at Sepang,
  Malaysia** (OpenF1 `circuit_key` 12 = Sepang, not Sakhir / key 63), so it never reuses the
  Sakhir layout.
- MultiViewer sometimes serves an **older layout** for a season; each file records the actual
  `layout_year` returned.
- Official F1 **S1/S2/S3 timing-sector boundaries are not published** by these sources and are
  always labelled unavailable; marshal sectors are shown as marshal sectors, never as official
  timing sectors.

## Attribution shown in the app

OpenStreetMap-derived maps display "© OpenStreetMap contributors"; the Madrid outline displays
"© bacinger/f1-circuits (MIT)"; MultiViewer-derived maps name MultiViewer as the provider in the
circuit's provenance panel.
