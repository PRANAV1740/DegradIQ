"""Race-context analysis page — offline, engine-driven, from tracked data.

For a selected 2024 race (from the tracked multi-event replay fixture) this page shows
the race's first-stop *context* and an event-level **predicted-vs-actual** comparison
scored with the shared ``PredictionEngine``:

  * Actual first-stop lap per driver and the field distribution.
  * Starting-compound mix.
  * Grid position vs actual first-stop lap.
  * Model predicted first-stop lap vs actual (per driver) and the event MAE.

Honest scope: weather, safety-car/VSC and traffic/gap context are **not** in the
certified dataset (and can't be re-derived — FastF1 timing host returns 403), so this
page reports the context available from lap/tyre/grid/compound data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.prediction_engine import load_default_engine

FIXTURE = (Path(__file__).resolve().parent / "reports" / "experiments" / "first_stop_v2"
           / "replay_events_2024.csv")

st.set_page_config(page_title="DegradIQ Race Context", page_icon="📊", layout="wide")
st.title("DegradIQ Race Context")
st.caption("First-stop context and predicted-vs-actual for recorded 2024 races, scored "
           "with the shared prediction engine. Weather / safety-car / traffic context is "
           "not available in the dataset and is not shown.")
st.info("**Retrospective / in-sample view.** These 2024 races are inside the served "
        "model's 2022–2024 training window, so the predicted-vs-actual shown here is a "
        "fit on data the model has already seen — **not** out-of-sample accuracy. For "
        "honest generalisation (unseen circuits/seasons), use the First-Stop page's "
        "Model Evaluation section and `PHASE3_TIMING.md`.")

engine = load_default_engine()


def _predicted_stop_lap(rows: pd.DataFrame) -> int | None:
    """Model's predicted first-stop lap for one driver: median of the survival curve."""
    rows = rows.sort_values("LapNumber")
    hazards = engine.score_rows(rows)
    laps = rows["LapNumber"].astype(int).to_numpy()
    surv = np.cumprod(1.0 - np.asarray(hazards, dtype=float))
    cum = 1.0 - surv
    idx = np.searchsorted(cum, 0.5)
    if idx >= len(laps):
        return int(laps[-1]) + 1
    return int(laps[idx]) + 1


if engine is None:
    st.error("No model is loaded; cannot compute predictions.")
elif not FIXTURE.exists():
    st.info("Race data fixture not found.")
else:
    data = pd.read_csv(FIXTURE)
    event = st.selectbox("Race", sorted(data["EventName"].unique()))
    ev = data[data["EventName"] == event]

    # Per-driver actual + predicted first-stop lap.
    recs = []
    for drv, g in ev.groupby("Driver"):
        pos = g[g["StopNextLap"] == 1]
        if pos.empty:
            continue
        actual = int(pos["LapNumber"].iloc[0]) + 1
        first = g.sort_values("LapNumber").iloc[0]
        try:
            pred = _predicted_stop_lap(g)
        except Exception:
            pred = None
        recs.append({"Driver": drv, "Grid": first.get("GridPosition"),
                     "StartCompound": str(first.get("Compound", "")).upper(),
                     "ActualStopLap": actual, "PredictedStopLap": pred})
    drivers = pd.DataFrame(recs)

    if drivers.empty:
        st.info("No drivers with an observed first stop in this race.")
    else:
        paired = drivers.dropna(subset=["PredictedStopLap"])
        mae = float(np.mean(np.abs(paired["PredictedStopLap"] - paired["ActualStopLap"]))) if len(paired) else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Drivers (with a first stop)", len(drivers))
        c2.metric("Median actual first-stop lap", int(drivers["ActualStopLap"].median()))
        c3.metric("Pred-vs-actual MAE (in-sample)", f"{mae:.1f} laps" if mae is not None else "–")

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            # Predicted vs actual.
            if len(paired):
                lo = int(min(paired["ActualStopLap"].min(), paired["PredictedStopLap"].min()))
                hi = int(max(paired["ActualStopLap"].max(), paired["PredictedStopLap"].max()))
                fig = px.scatter(paired, x="ActualStopLap", y="PredictedStopLap",
                                 text="Driver",
                                 title="Predicted vs actual first-stop lap (retrospective / in-sample)")
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         name="perfect", line=dict(dash="dash")))
                fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
                st.plotly_chart(fig, use_container_width=True)

            cc1, cc2 = st.columns(2)
            # Actual stop-lap distribution.
            cc1.plotly_chart(px.histogram(drivers, x="ActualStopLap", nbins=15,
                             title="Actual first-stop lap distribution"), use_container_width=True)
            # Starting compound mix.
            mix = drivers["StartCompound"].value_counts().reset_index()
            mix.columns = ["Compound", "Count"]
            cc2.plotly_chart(px.bar(mix, x="Compound", y="Count",
                             title="Starting compound mix"), use_container_width=True)

            # Grid vs actual stop lap.
            gp = drivers.dropna(subset=["Grid"])
            if len(gp):
                st.plotly_chart(px.scatter(gp, x="Grid", y="ActualStopLap", text="Driver",
                                title="Grid position vs actual first-stop lap")
                                .update_traces(textposition="top center"),
                                use_container_width=True)
        except Exception as exc:
            st.warning(f"Charts unavailable: {type(exc).__name__}")

        st.dataframe(drivers, use_container_width=True, hide_index=True)
        st.warning("These 2024 races are inside the served model's training window "
                   "(2022–2024), so predicted-vs-actual here is **in-sample fit**, not "
                   "out-of-sample performance. For honest generalisation numbers (unseen "
                   "circuits/seasons), see the Model Evaluation section and "
                   "`PHASE3_TIMING.md`.")
        st.caption("Predicted lap uses the shared PredictionEngine (per-lap hazard → "
                   "survival median). Context limitations: no weather, safety-car/VSC or "
                   "traffic data in the dataset.")
