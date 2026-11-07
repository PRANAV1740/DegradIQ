# Circuit Intelligence — geometry coverage report

_Generated 2026-11-07_

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
| jeddah | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (284m, gap 0m): length 283m outside expected window; raceway_ways loop (5521m, gap 173m): closure gap 173m exceeds tolerance 76m; raceway_ways loop (1693m, gap 1037m): closure gap 1037m exceeds tolerance 40m) |
| albert-park | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (5314m, gap 0m): loop closed only via 29 coordinate joins, not OSM node topology) |
| suzuka | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (6702m, gap 863m): closure gap 863m exceeds tolerance 59m; raceway_ways loop (1267m, gap 0m): length 1266m outside expected window; raceway_ways loop (480m, gap 0m): length 479m outside expected window; raceway_ways loop (165m, gap 0m): length 165m outside expected window) |
| shanghai | PARTIAL | osm_centreline | 5465 m loop |
| miami | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5691m, gap 718m): closure gap 718m exceeds tolerance 45m; raceway_ways loop (742m, gap 627m): closure gap 627m exceeds tolerance 40m) |
| imola | PARTIAL | osm_centreline | 4910 m loop |
| monaco | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (3305m, gap 24m): loop closed only via 38 coordinate joins, not OSM node topology; raceway_ways loop (251m, gap 0m): length 251m outside expected window; raceway_ways loop (408m, gap 324m): closure gap 324m exceeds tolerance 40m; raceway_ways loop (234m, gap 200m): closure gap 200m exceeds tolerance 40m) |
| montreal | PARTIAL | osm_centreline | 4367 m loop |
| barcelona | VERIFIED | fastlap_trace |  |
| madring | UNAVAILABLE | — | FastF1 not accessible (DataNotLoadedError; e.g. 403/not cached); no validated closed circuit loop in OpenStreetMap (raceway_ways loop (2187m, gap 579m): closure gap 579m exceeds tolerance 40m) |
| red-bull-ring | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (881m, gap 0m): length 880m outside expected window; relation loop (370m, gap 0m): length 370m outside expected window; relation loop (150m, gap 0m): length 149m outside expected window; relation loop (113m, gap 0m): length 112m outside expected window) |
| silverstone | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (9634m, gap 330m): loop closed only via 44 coordinate joins, not OSM node topology; relation loop (480m, gap 0m): loop closed only via 1 coordinate joins, not OSM node topology; relation loop (311m, gap 0m): loop closed only via 1 coordinate joins, not OSM node topology; raceway_ways loop (7016m, gap 786m): closure gap 786m exceeds tolerance 52m) |
| spa | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (7239m, gap 232m): closure gap 232m exceeds tolerance 62m; raceway_ways loop (1089m, gap 0m): length 1088m outside expected window; raceway_ways loop (1203m, gap 1163m): closure gap 1163m exceeds tolerance 40m; raceway_ways loop (720m, gap 470m): closure gap 470m exceeds tolerance 40m) |
| hungaroring | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (5534m, gap 0m): loop closed only via 11 coordinate joins, not OSM node topology; raceway_ways loop (5440m, gap 669m): closure gap 669m exceeds tolerance 40m; raceway_ways loop (276m, gap 0m): length 276m outside expected window; raceway_ways loop (421m, gap 0m): length 421m outside expected window) |
| zandvoort | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (156m, gap 119m): closure gap 119m exceeds tolerance 40m; raceway_ways loop (5016m, gap 459m): closure gap 459m exceeds tolerance 40m) |
| monza | VERIFIED | fastlap_trace |  |
| baku | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (relation loop (5975m, gap 0m): loop closed only via 49 coordinate joins, not OSM node topology) |
| marina-bay | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (1223m, gap 575m): closure gap 575m exceeds tolerance 40m) |
| cota | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (5785m, gap 271m): closure gap 271m exceeds tolerance 55m; raceway_ways loop (247m, gap 202m): closure gap 202m exceeds tolerance 40m; raceway_ways loop (950m, gap 718m): closure gap 718m exceeds tolerance 40m; raceway_ways loop (981m, gap 0m): length 981m outside expected window) |
| mexico | PARTIAL | osm_centreline | 3552 m loop |
| interlagos | PARTIAL | osm_centreline | 4356 m loop |
| las-vegas | UNAVAILABLE | — | no validated closed circuit loop in OpenStreetMap (raceway_ways loop (863m, gap 355m): closure gap 355m exceeds tolerance 40m) |
| lusail | PARTIAL | osm_centreline | 5472 m loop |
| yas-marina | PARTIAL | osm_centreline | 5383 m loop |

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
| 20 | São Paulo Grand Prix | interlagos | PARTIAL | osm_centreline |
| 21 | Las Vegas Grand Prix | las-vegas | UNAVAILABLE | — |
| 22 | Qatar Grand Prix | lusail | PARTIAL | osm_centreline |
| 23 | Abu Dhabi Grand Prix | yas-marina | PARTIAL | osm_centreline |

## Geometry sources & licensing

- **FastF1** racing-line traces: derived from F1 timing/telemetry for research use. Only the normalised outline and corner numbers are packaged; raw telemetry is not redistributed.
- **OpenStreetMap** centrelines: © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright). The derived outlines are an ODbL Derivative Database (share-alike applies) — see `OSM_ATTRIBUTION.md`. Each OSM circuit file records way ids, source URLs, the Overpass query and retrieval time.
