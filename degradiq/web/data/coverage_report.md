# Circuit Intelligence — geometry coverage report

_Generated 2026-11-22_

Four classifications — a PARTIAL map is never treated as complete:

- **VERIFIED** — FastF1 racing-line trace with real corner numbers & marshal sectors.
- **PARTIAL** — validated OpenStreetMap closed-loop centreline outline; no corners/sectors.
- **UNAVAILABLE** — no cached telemetry and no usable OSM circuit loop; honest empty state.
- **FAILED** — a source was attempted but acquisition/validation failed.

- Supported events: **23**
- Circuit configurations: **25**
- VERIFIED: **3**  ·  PARTIAL: **15**  ·  UNAVAILABLE: **7**  ·  FAILED: **0**

## Per-circuit status

| Circuit | Status | Geometry | Detail |
|---|---|---|---|
| bahrain | VERIFIED | fastlap_trace |  |
| jeddah | UNAVAILABLE | — | identity: no osm_identity configured for this circuit |
| albert-park | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| suzuka | PARTIAL | osm_centreline | 6702 m loop |
| shanghai | PARTIAL | osm_centreline | 5465 m loop |
| miami | PARTIAL | osm_centreline | 5691 m loop |
| imola | PARTIAL | osm_centreline | 4910 m loop |
| monaco | UNAVAILABLE | — | no validated circuit geometry in OpenStreetMap (raceway_ways loop (251m, gap 0m): length 251m outside expected window; raceway_ways loop (408m, gap 324m): not node-topology closed (node joins 1, coord joins 0); raceway_ways loop (234m, gap 200m): not node-topology closed (node joins 0, coord joins 0)) |
| montreal | PARTIAL | osm_centreline | 4367 m loop |
| barcelona | VERIFIED | fastlap_trace |  |
| madring | UNAVAILABLE | — | FastF1 not accessible (DataNotLoadedError; e.g. 403/not cached); no validated circuit geometry in OpenStreetMap (raceway_ways loop (2187m, gap 579m): not node-topology closed (node joins 0, coord joins 0)) |
| red-bull-ring | PARTIAL | osm_centreline | 4587 m loop |
| silverstone | PARTIAL | osm_centreline | 7016 m loop |
| spa | PARTIAL | osm_centreline | 7239 m loop |
| hungaroring | PARTIAL | osm_centreline | 5440 m loop |
| zandvoort | PARTIAL | osm_centreline | 5016 m loop |
| monza | VERIFIED | fastlap_trace |  |
| baku | UNAVAILABLE | — | no validated circuit geometry in OpenStreetMap (no candidate loops) |
| marina-bay | UNAVAILABLE | — | no validated circuit geometry in OpenStreetMap (raceway_ways loop (1223m, gap 575m): not node-topology closed (node joins 3, coord joins 0)) |
| cota | PARTIAL | osm_centreline | 5785 m loop |
| mexico | PARTIAL | osm_centreline | 3552 m loop |
| interlagos | PARTIAL | osm_centreline | 5692 m loop |
| las-vegas | UNAVAILABLE | — | no validated circuit geometry in OpenStreetMap (raceway_ways loop (863m, gap 355m): not node-topology closed (node joins 5, coord joins 0)) |
| lusail | PARTIAL | osm_centreline | 5434 m loop |
| yas-marina | PARTIAL | osm_centreline | 5597 m loop |

## Per-event coverage (2026 calendar)

| Round | Event | Circuit | Status | Geometry |
|---|---|---|---|---|
| 1 | Australian Grand Prix | albert-park | UNAVAILABLE | — |
| 2 | Chinese Grand Prix | shanghai | PARTIAL | osm_centreline |
| 3 | Japanese Grand Prix | suzuka | PARTIAL | osm_centreline |
| 4 | Miami Grand Prix | miami | PARTIAL | osm_centreline |
| 5 | Canadian Grand Prix | montreal | PARTIAL | osm_centreline |
| 6 | Monaco Grand Prix | monaco | UNAVAILABLE | — |
| 7 | Barcelona Grand Prix | barcelona | VERIFIED | fastlap_trace |
| 8 | Austrian Grand Prix | red-bull-ring | PARTIAL | osm_centreline |
| 9 | British Grand Prix | silverstone | PARTIAL | osm_centreline |
| 10 | Belgian Grand Prix | spa | PARTIAL | osm_centreline |
| 11 | Hungarian Grand Prix | hungaroring | PARTIAL | osm_centreline |
| 12 | Dutch Grand Prix | zandvoort | PARTIAL | osm_centreline |
| 13 | Italian Grand Prix | monza | VERIFIED | fastlap_trace |
| 14 | Spanish Grand Prix | madring | UNAVAILABLE | — |
| 15 | Azerbaijan Grand Prix | baku | UNAVAILABLE | — |
| 16 | Bahrain Grand Prix | bahrain | VERIFIED | fastlap_trace |
| 17 | Singapore Grand Prix | marina-bay | UNAVAILABLE | — |
| 18 | United States Grand Prix | cota | PARTIAL | osm_centreline |
| 19 | Mexico City Grand Prix | mexico | PARTIAL | osm_centreline |
| 20 | São Paulo Grand Prix | interlagos | PARTIAL | osm_centreline |
| 21 | Las Vegas Grand Prix | las-vegas | UNAVAILABLE | — |
| 22 | Qatar Grand Prix | lusail | PARTIAL | osm_centreline |
| 23 | Abu Dhabi Grand Prix | yas-marina | PARTIAL | osm_centreline |

## Geometry sources & licensing

- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised outline and corner numbers are packaged; raw telemetry is not redistributed.
- **OpenStreetMap** centrelines: © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). The derived outlines are an ODbL Derivative Database (share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, the Overpass query and retrieval time.
