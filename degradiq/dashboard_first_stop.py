"""Streamlit first-stop dashboard, driven by the validated PredictionEngine.

Offline and engine-backed: it loads the model once through ``PredictionEngine``
(no network, no retraining) and surfaces (1) model status + provenance, (2) a
first-stop prediction section with per-lap and cumulative probabilities and the
10-90 interval WIDTH, and (3) a model-evaluation section read from the tracked
generalisation summary. No fake predictions: if the model artifact is absent the
page says so instead of inventing numbers.

Run from the repository root:

    streamlit run degradiq/dashboard_first_stop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.config import YEAR, FULL_SEASON_DRIVERS, DRIVER_TEAM_2026
from degradiq.prediction_engine import load_default_engine, PredictionEngine
from degradiq.live_adapter import UnavailableLiveSource

SUMMARY_PATH = (Path(__file__).resolve().parent / "reports" / "experiments"
                / "first_stop_v2" / "generalization_summary.json")

st.set_page_config(page_title="DegradIQ First-Stop", page_icon="🏁", layout="wide")
st.title("DegradIQ First-Stop")
st.caption(f"{YEAR} first genuine strategic pit-stop forecasts from the validated "
           "discrete-hazard engine. Probabilistic risk, not a guaranteed strategy.")

# Honest data-source indicator (Phase VI): no real-time feed is available, so the
# page offers pre-race forecasts and historical replay, never invented live timing.
_live_status = UnavailableLiveSource().status()
if not _live_status.available:
    st.warning(f"**Live data: {_live_status.source}.** {_live_status.reason}")


@st.cache_resource(show_spinner=False)
def get_engine() -> PredictionEngine | None:
    return load_default_engine()


engine = get_engine()

# ---- Sidebar: model status + provenance -----------------------------------
with st.sidebar:
    st.header("Model status")
    if engine is None:
        st.error("No model artifact is loaded. Train it with "
                 "`python -m degradiq.scripts.train_first_stop` and place the artifact "
                 "in `degradiq/artifacts/`.")
    else:
        prov = engine.provenance()
        st.success("Model ready")
        status_badge = {"current": "✅ current", "unversioned_legacy": "⚠️ legacy (unversioned)",
                        "stale": "⛔ stale"}.get(prov["label_version_status"], prov["label_version_status"])
        st.write(f"**Label version:** {status_badge}")
        st.write(f"**Architecture:** {prov.get('architecture')}")
        st.write(f"**Trained:** {prov.get('trained_at_utc') or 'unknown'}")
        st.write(f"**Seasons:** {prov.get('training_seasons') or 'unknown'}")
        st.caption(f"sha256 {(prov.get('model_sha256') or '')[:12]}…")

# ---- First-stop prediction -------------------------------------------------
st.subheader("First-stop prediction")
if engine is None:
    st.info("Load a model to generate forecasts.")
else:
    c1, c2, c3 = st.columns([2, 1, 1])
    event_name = c1.text_input("Event", value="United States Grand Prix")
    race_laps = int(c2.number_input("Race laps", min_value=5, max_value=100, value=56))
    mode = c3.selectbox("Feature mode", ["pre-race", "live"], index=0)

    all_drivers = sorted(FULL_SEASON_DRIVERS)
    picked = st.multiselect("Drivers", all_drivers,
                            default=[d for d in ("VER", "NOR", "LEC", "HAM") if d in all_drivers])
    compound = st.selectbox("Starting compound", ["SOFT", "MEDIUM", "HARD"], index=1)

    if st.button("Forecast first stops", type="primary") and picked:
        drivers = [{"driver": code, "grid_position": i + 1, "starting_compound": compound,
                    "starting_tyre_age": 1,
                    "team": DRIVER_TEAM_2026.get(code, "Unknown")}
                   for i, code in enumerate(picked)]
        try:
            result = engine.predict(event_name, race_laps, drivers, mode=mode,
                                    context_mode="historical")
        except Exception as exc:  # surfaced, never silently fabricated
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            result = None

        if result and result.get("predictions"):
            rows = [{
                "Driver": p["driver"],
                "P(stop before finish)": round(p.get("probability_stop_before_finish", 0.0), 3),
                "Most likely lap": p.get("most_likely_stop_lap"),
                "Median lap (if stops)": p.get("median_stop_lap_if_stops"),
                "10-90 window": (f"{p['likely_window_if_stops']['from_lap']}–"
                                 f"{p['likely_window_if_stops']['to_lap']}"
                                 if p.get("likely_window_if_stops") else None),
                "Interval width (laps)": p.get("interval_width_if_stops"),
            } for p in result["predictions"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("The 10-90 window is wide by construction (typically ~15-25 laps): "
                       "treat it as an uncertainty band, not a precise lap call. Feature mode "
                       f"**{mode}**, context **historical**.")

            # Cumulative probability of having stopped by each lap.
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                for p in result["predictions"]:
                    cum = p.get("cumulative_probabilities") or []
                    if cum:
                        fig.add_trace(go.Scatter(
                            x=[c["lap"] for c in cum],
                            y=[c["probability_stopped_by_lap"] for c in cum],
                            mode="lines", name=p["driver"]))
                fig.update_layout(title="Cumulative probability of first stop by lap",
                                  xaxis_title="Lap", yaxis_title="P(stopped by lap)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# ---- Historical replay (point-in-time) -------------------------------------
REPLAY_PATH = (Path(__file__).resolve().parent / "reports" / "experiments"
               / "first_stop_v2" / "replay_sample_spanish_2024.csv")
st.subheader("Historical replay (point-in-time)")
st.caption("Spanish Grand Prix 2024. At each replay lap the model sees only the "
           "feature rows available up to that lap; the actual first stop is revealed "
           "for comparison. Scored with the loaded model via its per-lap hazard.")
if engine is None:
    st.info("Load a model to run replay.")
elif not REPLAY_PATH.exists():
    st.info("Replay sample not found.")
else:
    import numpy as np
    import plotly.graph_objects as go

    replay = pd.read_csv(REPLAY_PATH)
    drv = st.selectbox("Replay driver", sorted(replay["Driver"].unique()))
    d = replay[replay["Driver"] == drv].sort_values("LapNumber").reset_index(drop=True)
    # Actual first stop = the predicting lap with StopNextLap==1, plus one.
    pos = d[d["StopNextLap"] == 1]
    actual_stop_lap = int(pos["LapNumber"].iloc[0]) + 1 if not pos.empty else None

    try:
        hazards = engine.score_rows(d)
    except Exception as exc:
        st.error(f"Replay scoring failed: {type(exc).__name__}: {exc}")
        hazards = None

    if hazards is not None and len(d):
        laps = d["LapNumber"].astype(int).to_numpy()
        surv = np.cumprod(1.0 - np.asarray(hazards, dtype=float))
        cum_stopped = 1.0 - surv  # P(stopped by end of each lap)
        max_lap = int(laps.max())
        replay_lap = st.slider("Replay lap", int(laps.min()), max_lap, int(laps.min()))

        shown = laps <= replay_lap
        c1, c2, c3 = st.columns(3)
        c1.metric("Replay lap", replay_lap)
        c2.metric("Model P(stopped by now)", f"{cum_stopped[shown][-1]:.2f}" if shown.any() else "–")
        c3.metric("Actual first stop lap",
                  str(actual_stop_lap) if (actual_stop_lap is not None and actual_stop_lap <= replay_lap)
                  else "not yet (hidden)")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=laps[shown], y=np.asarray(hazards)[shown],
                             name="Per-lap hazard", marker_color="#8899ff"))
        fig.add_trace(go.Scatter(x=laps[shown], y=cum_stopped[shown], yaxis="y2",
                                 mode="lines+markers", name="Cumulative P(stopped)",
                                 line=dict(color="#ff6b6b")))
        if actual_stop_lap is not None and actual_stop_lap <= replay_lap:
            fig.add_vline(x=actual_stop_lap, line_dash="dash", line_color="#2ecc71",
                          annotation_text="actual stop")
        fig.update_layout(
            title=f"{drv}: per-lap hazard and cumulative stop probability (through lap {replay_lap})",
            xaxis_title="Lap", yaxis=dict(title="Per-lap hazard"),
            yaxis2=dict(title="Cumulative P(stopped)", overlaying="y", side="right", range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Point-in-time: only laps ≤ the replay lap inform the displayed curve. "
                   "The green line marks the actual first stop once reached.")

# ---- Model evaluation (from tracked generalisation summary) ----------------
st.subheader("Model evaluation (independent validation)")
if not SUMMARY_PATH.exists():
    st.info("Generalisation summary not found. Run "
            "`python -m degradiq.scripts.validate_generalization`.")
else:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = []
    for r in summary.get("temporal_holdout", []):
        c = r.get("candidate", {})
        rows.append({
            "Regime": r.get("tag", "").replace("_", " "),
            "MAE": round(c.get("first_stop_lap_mae"), 2) if c.get("first_stop_lap_mae") is not None else None,
            "Baseline MAE": round(r.get("baseline_historical_median_mae"), 2) if r.get("baseline_historical_median_mae") is not None else None,
            "AUC": round(c.get("roc_auc"), 3) if c.get("roc_auc") is not None else None,
            "Brier": round(c.get("brier_score"), 4) if c.get("brier_score") is not None else None,
            "10-90 coverage": round(c.get("interval_coverage_10_90"), 3) if c.get("interval_coverage_10_90") is not None else None,
            "10-90 width": round(c.get("interval_width_10_90_mean"), 1) if c.get("interval_width_10_90_mean") is not None else None,
        })
    loco = summary.get("leave_one_circuit_out")
    if loco:
        m = loco.get("macro_mean_over_circuits", {})
        rows.append({
            "Regime": f"LOCO macro ({loco.get('n_circuits')} unseen circuits)",
            "MAE": round(m.get("first_stop_lap_mae"), 2) if m.get("first_stop_lap_mae") is not None else None,
            "Baseline MAE": None,
            "AUC": round(m.get("roc_auc"), 3) if m.get("roc_auc") is not None else None,
            "Brier": round(m.get("brier_score"), 4) if m.get("brier_score") is not None else None,
            "10-90 coverage": round(m.get("interval_coverage_10_90"), 3) if m.get("interval_coverage_10_90") is not None else None,
            "10-90 width": round(m.get("interval_width_10_90_mean"), 1) if m.get("interval_width_10_90_mean") is not None else None,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if loco and loco.get("timing_mae_vs_baseline"):
        t = loco["timing_mae_vs_baseline"]
        st.caption(
            f"Honest reading: classification generalises to unseen circuits (LOCO AUC ≈ "
            f"seen-circuit holdout), but the pre-race timing MAE matches or beats the "
            f"historical-median baseline in only {t['candidate_le_baseline']}/"
            f"{t['circuits_evaluated']} unseen circuits, and the ~{rows[-1].get('10-90 width')}-lap "
            f"intervals are conservative, not sharp. These are development / partial-validation "
            f"results, not a blind-season test.")
