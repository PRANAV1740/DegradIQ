# Circuit Intelligence — geometry coverage report

_Generated 2026-11-07_

Four classifications — a PARTIAL map is never treated as complete:

- **VERIFIED** — FastF1 racing-line trace with real corner numbers & marshal sectors.
- **PARTIAL** — validated OpenStreetMap closed-loop centreline outline; no corners/sectors.
- **UNAVAILABLE** — no cached telemetry and no usable OSM circuit loop; honest empty state.
- **FAILED** — a source was attempted but acquisition/validation failed.

- Supported events: **23**
- Circuit configurations: **25**
- VERIFIED: **3**  ·  PARTIAL: **4**  ·  UNAVAILABLE: **18**  ·  FAILED: **0**

## Per-circuit status

| Circuit | Status | Geometry | Detail |
|---|---|---|---|
| bahrain | VERIFIED | fastlap_trace |  |
| jeddah | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (284m, gap 0m): length 283m outside expected window; raceway_ways loop (5521m, gap 173m): not node-topology closed (node joins 11, coord joins 0); raceway_ways loop (1693m, gap 1037m): not node-topology closed (node joins 1, coord joins 0)) |
| albert-park | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| suzuka | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (6702m, gap 863m): not node-topology closed (node joins 40, coord joins 0); raceway_ways loop (1267m, gap 0m): length 1266m outside expected window; raceway_ways loop (480m, gap 0m): length 479m outside expected window; raceway_ways loop (165m, gap 0m): length 165m outside expected window) |
| shanghai | PARTIAL | osm_centreline | 5465 m loop |
| miami | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5691m, gap 718m): not node-topology closed (node joins 6, coord joins 0); raceway_ways loop (742m, gap 627m): not node-topology closed (node joins 0, coord joins 0)) |
| imola | PARTIAL | osm_centreline | 4910 m loop |
| monaco | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (251m, gap 0m): length 251m outside expected window; raceway_ways loop (408m, gap 324m): not node-topology closed (node joins 1, coord joins 0); raceway_ways loop (234m, gap 200m): not node-topology closed (node joins 0, coord joins 0)) |
| montreal | PARTIAL | osm_centreline | 4367 m loop |
| barcelona | VERIFIED | fastlap_trace |  |
| madring | UNAVAILABLE | — | FastF1 not accessible (DataNotLoadedError; e.g. 403/not cached); no validated closed circuit loop in OpenStreetMap (raceway_ways loop (2187m, gap 579m): not node-topology closed (node joins 0, coord joins 0)) |
| red-bull-ring | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (305m, gap 44m): not node-topology closed (node joins 2, coord joins 0); raceway_ways loop (229m, gap 116m): not node-topology closed (node joins 2, coord joins 0); raceway_ways loop (4587m, gap 230m): not node-topology closed (node joins 12, coord joins 0); raceway_ways loop (1018m, gap 23m): not node-topology closed (node joins 2, coord joins 0)) |
| silverstone | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (9634m, gap 330m): closure gap 330m exceeds absolute tolerance 40m; relation loop (480m, gap 0m): length 480m outside expected window; relation loop (311m, gap 0m): length 311m outside expected window; raceway_ways loop (7016m, gap 786m): not node-topology closed (node joins 27, coord joins 0)) |
| spa | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (7239m, gap 232m): not node-topology closed (node joins 30, coord joins 0); raceway_ways loop (1089m, gap 0m): length 1088m outside expected window; raceway_ways loop (1203m, gap 1163m): not node-topology closed (node joins 0, coord joins 0); raceway_ways loop (720m, gap 470m): not node-topology closed (node joins 0, coord joins 0)) |
| hungaroring | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5440m, gap 669m): not node-topology closed (node joins 2, coord joins 0); raceway_ways loop (276m, gap 0m): length 276m outside expected window; raceway_ways loop (421m, gap 0m): length 421m outside expected window; raceway_ways loop (167m, gap 6m): not node-topology closed (node joins 1, coord joins 0)) |
| zandvoort | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (156m, gap 119m): not node-topology closed (node joins 0, coord joins 0); raceway_ways loop (5016m, gap 459m): not node-topology closed (node joins 24, coord joins 0)) |
| monza | VERIFIED | fastlap_trace |  |
| baku | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (no candidate loops) |
| marina-bay | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (1223m, gap 575m): not node-topology closed (node joins 3, coord joins 0)) |
| cota | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5785m, gap 271m): not node-topology closed (node joins 32, coord joins 0); raceway_ways loop (247m, gap 202m): not node-topology closed (node joins 0, coord joins 0); raceway_ways loop (950m, gap 718m): not node-topology closed (node joins 0, coord joins 0); raceway_ways loop (981m, gap 0m): length 981m outside expected window) |
| mexico | PARTIAL | osm_centreline | 3552 m loop |
| interlagos | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5692m, gap 657m): not node-topology closed (node joins 25, coord joins 0); raceway_ways loop (736m, gap 80m): not node-topology closed (node joins 4, coord joins 0); raceway_ways loop (443m, gap 283m): not node-topology closed (node joins 2, coord joins 0); raceway_ways loop (155m, gap 133m): not node-topology closed (node joins 0, coord joins 0)) |
| las-vegas | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (863m, gap 355m): not node-topology closed (node joins 5, coord joins 0)) |
| lusail | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (6668m, gap 993m): closure gap 993m exceeds absolute tolerance 40m; raceway_ways loop (5434m, gap 26m): not node-topology closed (node joins 3, coord joins 0); raceway_ways loop (529m, gap 433m): not node-topology closed (node joins 0, coord joins 0); raceway_ways loop (746m, gap 712m): not node-topology closed (node joins 0, coord joins 0)) |
| yas-marina | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5597m, gap 224m): not node-topology closed (node joins 6, coord joins 0); raceway_ways loop (1130m, gap 620m): not node-topology closed (node joins 2, coord joins 0); raceway_ways loop (1023m, gap 749m): not node-topology closed (node joins 1, coord joins 0); raceway_ways loop (237m, gap 218m): not node-topology closed (node joins 0, coord joins 0)) |

## Per-event coverage (2026 calendar)

| Round | Event | Circuit | Status | Geometry |
|---|---|---|---|---|
| 1 | Australian Grand Prix | albert-park | UNAVAILABLE | — |
| 2 | Chinese Grand Prix | shanghai | PARTIAL | osm_centreline |
| 3 | Japanese Grand Prix | suzuka | UNAVAILABLE | — |
| 4 | Miami Grand Prix | miami | UNAVAILABLE | — |
| 5 | Canadian Grand Prix | montreal | PARTIAL | osm_centreline |
| 6 | Monaco Grand Prix | monaco | UNAVAILABLE | — |
| 7 | Barcelona Grand Prix | barcelona | VERIFIED | fastlap_trace |
| 8 | Austrian Grand Prix | red-bull-ring | UNAVAILABLE | — |
| 9 | British Grand Prix | silverstone | UNAVAILABLE | — |
| 10 | Belgian Grand Prix | spa | UNAVAILABLE | — |
| 11 | Hungarian Grand Prix | hungaroring | UNAVAILABLE | — |
| 12 | Dutch Grand Prix | zandvoort | UNAVAILABLE | — |
| 13 | Italian Grand Prix | monza | VERIFIED | fastlap_trace |
| 14 | Spanish Grand Prix | madring | UNAVAILABLE | — |
| 15 | Azerbaijan Grand Prix | baku | UNAVAILABLE | — |
| 16 | Bahrain Grand Prix | bahrain | VERIFIED | fastlap_trace |
| 17 | Singapore Grand Prix | marina-bay | UNAVAILABLE | — |
| 18 | United States Grand Prix | cota | UNAVAILABLE | — |
| 19 | Mexico City Grand Prix | mexico | PARTIAL | osm_centreline |
| 20 | São Paulo Grand Prix | interlagos | UNAVAILABLE | — |
| 21 | Las Vegas Grand Prix | las-vegas | UNAVAILABLE | — |
| 22 | Qatar Grand Prix | lusail | UNAVAILABLE | — |
| 23 | Abu Dhabi Grand Prix | yas-marina | UNAVAILABLE | — |

## Geometry sources & licensing

- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised outline and corner numbers are packaged; raw telemetry is not redistributed.
- **OpenStreetMap** centrelines: © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). The derived outlines are an ODbL Derivative Database (share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, the Overpass query and retrieval time.
