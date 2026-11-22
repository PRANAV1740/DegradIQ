# Circuit Intelligence — geometry coverage report

_Generated 2026-11-22_

Four classifications — a PARTIAL map is never treated as complete:

- **VERIFIED** — FastF1 racing-line trace with real corner numbers & marshal sectors.
- **PARTIAL** — validated OpenStreetMap closed-loop centreline outline; no corners/sectors.
- **UNAVAILABLE** — no cached telemetry and no usable OSM circuit loop; honest empty state.
- **FAILED** — a source was attempted but acquisition/validation failed.

- Supported events: **23**
- Circuit configurations: **25**
- VERIFIED: **21**  ·  PARTIAL: **3**  ·  UNAVAILABLE: **1**  ·  FAILED: **0**

## Per-circuit status

| Circuit | Status | Geometry | Detail |
|---|---|---|---|
| bahrain | PARTIAL | osm_centreline | 4615 m loop |
| jeddah | UNAVAILABLE | — | identity: no osm_identity configured for this circuit |
| albert-park | VERIFIED | fastlap_trace |  |
| suzuka | VERIFIED | fastlap_trace |  |
| shanghai | VERIFIED | fastlap_trace |  |
| miami | VERIFIED | fastlap_trace |  |
| imola | PARTIAL | osm_centreline | 4910 m loop |
| monaco | VERIFIED | fastlap_trace |  |
| montreal | VERIFIED | fastlap_trace |  |
| barcelona | VERIFIED | fastlap_trace |  |
| madring | PARTIAL | osm_centreline | 5437 m loop |
| red-bull-ring | VERIFIED | fastlap_trace |  |
| silverstone | VERIFIED | fastlap_trace |  |
| spa | VERIFIED | fastlap_trace |  |
| hungaroring | VERIFIED | fastlap_trace |  |
| zandvoort | VERIFIED | fastlap_trace |  |
| monza | VERIFIED | fastlap_trace |  |
| baku | VERIFIED | fastlap_trace |  |
| marina-bay | VERIFIED | fastlap_trace |  |
| cota | VERIFIED | fastlap_trace |  |
| mexico | VERIFIED | fastlap_trace |  |
| interlagos | VERIFIED | fastlap_trace |  |
| las-vegas | VERIFIED | fastlap_trace |  |
| lusail | VERIFIED | fastlap_trace |  |
| yas-marina | VERIFIED | fastlap_trace |  |

## Per-event coverage (2026 calendar)

| Round | Event | Circuit | Status | Geometry |
|---|---|---|---|---|
| 1 | Australian Grand Prix | albert-park | VERIFIED | fastlap_trace |
| 2 | Chinese Grand Prix | shanghai | VERIFIED | fastlap_trace |
| 3 | Japanese Grand Prix | suzuka | VERIFIED | fastlap_trace |
| 4 | Miami Grand Prix | miami | VERIFIED | fastlap_trace |
| 5 | Canadian Grand Prix | montreal | VERIFIED | fastlap_trace |
| 6 | Monaco Grand Prix | monaco | VERIFIED | fastlap_trace |
| 7 | Barcelona Grand Prix | barcelona | VERIFIED | fastlap_trace |
| 8 | Austrian Grand Prix | red-bull-ring | VERIFIED | fastlap_trace |
| 9 | British Grand Prix | silverstone | VERIFIED | fastlap_trace |
| 10 | Belgian Grand Prix | spa | VERIFIED | fastlap_trace |
| 11 | Hungarian Grand Prix | hungaroring | VERIFIED | fastlap_trace |
| 12 | Dutch Grand Prix | zandvoort | VERIFIED | fastlap_trace |
| 13 | Italian Grand Prix | monza | VERIFIED | fastlap_trace |
| 14 | Spanish Grand Prix | madring | PARTIAL | osm_centreline |
| 15 | Azerbaijan Grand Prix | baku | VERIFIED | fastlap_trace |
| 16 | Bahrain Grand Prix | bahrain | PARTIAL | osm_centreline |
| 17 | Singapore Grand Prix | marina-bay | VERIFIED | fastlap_trace |
| 18 | United States Grand Prix | cota | VERIFIED | fastlap_trace |
| 19 | Mexico City Grand Prix | mexico | VERIFIED | fastlap_trace |
| 20 | São Paulo Grand Prix | interlagos | VERIFIED | fastlap_trace |
| 21 | Las Vegas Grand Prix | las-vegas | VERIFIED | fastlap_trace |
| 22 | Qatar Grand Prix | lusail | VERIFIED | fastlap_trace |
| 23 | Abu Dhabi Grand Prix | yas-marina | VERIFIED | fastlap_trace |

## Geometry sources & licensing

- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised outline and corner numbers are packaged; raw telemetry is not redistributed.
- **OpenStreetMap** centrelines: © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). The derived outlines are an ODbL Derivative Database (share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, the Overpass query and retrieval time.
