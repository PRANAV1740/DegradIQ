# Circuit Intelligence — geometry coverage report

_Generated 2026-11-06_

Four classifications — a PARTIAL map is never treated as complete:

- **VERIFIED** — FastF1 racing-line trace with real corner numbers & marshal sectors.
- **PARTIAL** — validated OpenStreetMap closed-loop centreline outline; no corners/sectors.
- **UNAVAILABLE** — no cached telemetry and no usable OSM circuit loop; honest empty state.
- **FAILED** — a source was attempted but acquisition/validation failed.

- Supported events: **23**
- Circuit configurations: **25**
- VERIFIED: **3**  ·  PARTIAL: **7**  ·  UNAVAILABLE: **15**  ·  FAILED: **0**

## Per-circuit status

| Circuit | Status | Geometry | Detail |
|---|---|---|---|
| bahrain | VERIFIED | fastlap_trace |  |
| jeddah | PARTIAL | osm_centreline | 5521 m loop |
| albert-park | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| suzuka | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (6702m, gap 863m): not closed; raceway_ways loop (1267m, gap 0m): length 1266m outside expected window; raceway_ways loop (480m, gap 0m): length 479m outside expected window; raceway_ways loop (165m, gap 0m): length 165m outside expected window) |
| shanghai | PARTIAL | osm_centreline | 5465 m loop |
| miami | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5691m, gap 718m): not closed; raceway_ways loop (742m, gap 627m): not closed) |
| imola | PARTIAL | osm_centreline | 4910 m loop |
| monaco | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (251m, gap 0m): length 251m outside expected window; raceway_ways loop (408m, gap 324m): not closed; raceway_ways loop (234m, gap 200m): length 234m outside expected window) |
| montreal | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| barcelona | VERIFIED | fastlap_trace |  |
| madring | UNAVAILABLE | — | FastF1 not accessible (DataNotLoadedError; e.g. 403/not cached); no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| red-bull-ring | PARTIAL | osm_centreline | 4587 m loop |
| silverstone | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (2638m, gap 0m): length 2638m outside expected window; raceway_ways loop (3153m, gap 301m): not closed; raceway_ways loop (483m, gap 415m): not closed; raceway_ways loop (2191m, gap 403m): not closed) |
| spa | PARTIAL | osm_centreline | 7239 m loop |
| hungaroring | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5440m, gap 669m): not closed; raceway_ways loop (276m, gap 0m): length 276m outside expected window; raceway_ways loop (421m, gap 0m): length 421m outside expected window; raceway_ways loop (172m, gap 0m): length 171m outside expected window) |
| zandvoort | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| monza | VERIFIED | fastlap_trace |  |
| baku | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| marina-bay | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (1223m, gap 575m): not closed) |
| cota | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5785m, gap 271m): not closed; raceway_ways loop (247m, gap 202m): length 247m outside expected window; raceway_ways loop (950m, gap 718m): not closed; raceway_ways loop (981m, gap 0m): length 981m outside expected window) |
| mexico | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (1035m, gap 173m): length 1034m outside expected window; raceway_ways loop (3850m, gap 504m): not closed; raceway_ways loop (507m, gap 223m): length 506m outside expected window; raceway_ways loop (927m, gap 596m): not closed) |
| interlagos | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5692m, gap 657m): not closed; raceway_ways loop (1553m, gap 120m): length 1553m outside expected window; raceway_ways loop (132m, gap 70m): length 131m outside expected window; raceway_ways loop (155m, gap 133m): length 155m outside expected window) |
| las-vegas | UNAVAILABLE | — | no highway=raceway ways or motorsport circuit relation in OpenStreetMap here (e.g. an un-mapped street circuit) |
| lusail | PARTIAL | osm_centreline | 5434 m loop |
| yas-marina | PARTIAL | osm_centreline | 5597 m loop |

## Per-event coverage (2026 calendar)

| Round | Event | Circuit | Status | Geometry |
|---|---|---|---|---|
| 1 | Australian Grand Prix | albert-park | UNAVAILABLE | — |
| 2 | Chinese Grand Prix | shanghai | PARTIAL | osm_centreline |
| 3 | Japanese Grand Prix | suzuka | UNAVAILABLE | — |
| 4 | Miami Grand Prix | miami | UNAVAILABLE | — |
| 5 | Canadian Grand Prix | montreal | UNAVAILABLE | — |
| 6 | Monaco Grand Prix | monaco | UNAVAILABLE | — |
| 7 | Barcelona Grand Prix | barcelona | VERIFIED | fastlap_trace |
| 8 | Austrian Grand Prix | red-bull-ring | PARTIAL | osm_centreline |
| 9 | British Grand Prix | silverstone | UNAVAILABLE | — |
| 10 | Belgian Grand Prix | spa | PARTIAL | osm_centreline |
| 11 | Hungarian Grand Prix | hungaroring | UNAVAILABLE | — |
| 12 | Dutch Grand Prix | zandvoort | UNAVAILABLE | — |
| 13 | Italian Grand Prix | monza | VERIFIED | fastlap_trace |
| 14 | Spanish Grand Prix | madring | UNAVAILABLE | — |
| 15 | Azerbaijan Grand Prix | baku | UNAVAILABLE | — |
| 16 | Bahrain Grand Prix | bahrain | VERIFIED | fastlap_trace |
| 17 | Singapore Grand Prix | marina-bay | UNAVAILABLE | — |
| 18 | United States Grand Prix | cota | UNAVAILABLE | — |
| 19 | Mexico City Grand Prix | mexico | UNAVAILABLE | — |
| 20 | São Paulo Grand Prix | interlagos | UNAVAILABLE | — |
| 21 | Las Vegas Grand Prix | las-vegas | UNAVAILABLE | — |
| 22 | Qatar Grand Prix | lusail | PARTIAL | osm_centreline |
| 23 | Abu Dhabi Grand Prix | yas-marina | PARTIAL | osm_centreline |

## Geometry sources & licensing

- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised outline and corner numbers are packaged; raw telemetry is not redistributed.
- **OpenStreetMap** centrelines: © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). The derived outlines are an ODbL Derivative Database (share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, the Overpass query and retrieval time.
