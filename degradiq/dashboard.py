"""Streamlit dashboard for event-by-event 2026 DegradIQ analysis."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Support `streamlit run degradiq/dashboard.py` from repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degradiq.config import YEAR, CACHE_DIR, PIT_THRESHOLDS_SECONDS
from degradiq.ingest import get_calendar, load_event_practice, load_race
from degradiq.validation import prepare_laps, validate_event
from degradiq.model import fit_hinge, predict, cumulative_degradation, recommend_pit_window

st.set_page_config(page_title="DegradIQ | 2026", page_icon="🏎️", layout="wide")
st.title("DegradIQ")
st.caption("2026 season tyre degradation isolation. Practice is the training set. Race laps are a holdout.")

@st.cache_data(ttl=3600, show_spinner=False)
def calendar_frame():
    return get_calendar(YEAR)

try:
    calendar = calendar_frame()
except Exception as exc:
    st.error(f"Could not retrieve the 2026 calendar from FastF1: {exc}")
    st.stop()

def event_label(row):
    date = row.get("EventDate")
    date_text = date.strftime("%d %b") if pd.notna(date) else "date unknown"
    return f"{row.get('RoundNumber', '')} | {row['EventName']} | {date_text}"

options = {event_label(row): row for _, row in calendar.iterrows()}
selected_label = st.selectbox("2026 event", list(options), index=0)
selected_event = options[selected_label]
event_id = selected_event.get("RoundNumber", selected_event["EventName"])
if pd.isna(event_id) or event_id == 0:
    event_id = selected_event["EventName"]

if selected_event["CalendarStatus"] == "Pending":
    st.info("This event is in the future. It remains pending until FastF1 has session data.")
    st.stop()

if st.button("Load practice and validate against race", type="primary"):
    with st.spinner("Loading event data from FastF1. First downloads can take time."):
        practice_raw, session_statuses = load_event_practice(YEAR, event_id, CACHE_DIR)
        race_raw, race_status = load_race(YEAR, event_id, CACHE_DIR)
        st.session_state["degradiq_payload"] = (practice_raw, race_raw, session_statuses, race_status, selected_label)

payload = st.session_state.get("degradiq_payload")
if not payload or payload[4] != selected_label:
    st.write("Choose a completed event and load its data to begin.")
    st.stop()

practice_raw, race_raw, session_statuses, race_status, _ = payload
st.subheader("Data coverage")
st.dataframe(pd.DataFrame(session_statuses), use_container_width=True, hide_index=True)
st.write(f"Race session: **{race_status}**")
if practice_raw.empty:
    st.warning("No eligible practice laps were loaded for full-season lineup drivers.")
    st.stop()

practice = prepare_laps(practice_raw)
race = prepare_laps(race_raw)
compounds = sorted(set(practice.get("Compound", [])) | set(race.get("Compound", [])))
if not compounds:
    st.warning("No compounds are available after the required filters.")
    st.stop()
compound = st.selectbox("Tyre compound", compounds)
model_info = fit_hinge(practice[practice["Compound"] == compound])
result_rows, details = validate_event(practice_raw, race_raw)
selected_result = details.get(compound, {})
mae = selected_result.get("mae_s_per_lap")
c1, c2, c3 = st.columns(3)
c1.metric("Eligible practice laps", f"{len(practice[practice['Compound'] == compound]):,}")
c2.metric("Eligible race laps", f"{selected_result.get('race_laps', 0):,}")
c3.metric("Race holdout MAE", f"{mae:.3f} s/lap" if mae is not None else "Not validated")
st.caption("MAE is computed after baseline-normalizing predicted and actual degradation within each race stint. It is not absolute lap-time error.")

if model_info.get("status") != "fit":
    st.warning(f"Model status: {model_info.get('status')}. No degradation curve or pit window is claimed.")
    st.dataframe(result_rows, use_container_width=True, hide_index=True)
    st.stop()

compound_practice = practice[practice["Compound"] == compound]
ages = sorted(compound_practice["TyreLife"].astype(float).unique())
x = pd.Series(ages, dtype=float)
raw = compound_practice.groupby("TyreLife")["LapTime_s"].median().sort_index()
corrected = compound_practice.groupby("TyreLife")["CorrectedTime_s"].median().sort_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw.index, y=raw.values, mode="lines+markers", name="Raw median"))
fig.add_trace(go.Scatter(x=corrected.index, y=corrected.values, mode="lines+markers", name="Corrected median"))
fig.update_layout(title="Before and after correction", xaxis_title="Tyre age (laps)", yaxis_title="Lap time (seconds)")
st.plotly_chart(fig, use_container_width=True)

grid = pd.Series(range(int(min(ages)), int(max(ages)) + 1), dtype=float)
preds = predict(model_info, grid.to_numpy())
curve = pd.DataFrame({"Tyre age": grid, "Modelled pace change (s)": preds - preds[0]})
st.plotly_chart(px.line(curve, x="Tyre age", y="Modelled pace change (s)", title="Practice-trained degradation curve"), use_container_width=True)

# Attribution is based on the selected longest practice stint and is shown as signed
# absolute contribution magnitudes, not as an overall season metric.
stint_keys = [c for c in ("EventName", "Session", "Driver", "Stint") if c in compound_practice.columns]
stints = list(compound_practice.groupby(stint_keys, dropna=False))
if stints:
    _, longest = max(stints, key=lambda pair: len(pair[1]))
    fuel = float(longest["FuelCorrection_s"].max() - longest["FuelCorrection_s"].min())
    track = float(longest["TrackCorrection_s"].max() - longest["TrackCorrection_s"].min())
    corrected_range = float(longest["CorrectedTime_s"].max() - longest["CorrectedTime_s"].min())
    raw_range = float(longest["LapTime_s"].max() - longest["LapTime_s"].min())
    attribution = pd.DataFrame({"Source": ["Fuel correction", "Track correction", "Corrected lap-time range"],
                                "Magnitude (s)": [fuel, track, corrected_range]})
    st.plotly_chart(px.bar(attribution, x="Source", y="Magnitude (s)", title="Longest practice stint: correction magnitudes"), use_container_width=True)
    st.caption("These are correction magnitudes, not causal percentages. Track evolution can legitimately be zero when the field trend is not faster.")

window = recommend_pit_window(model_info, max_age=max(60, int(max(ages))), thresholds=PIT_THRESHOLDS_SECONDS)
st.subheader("Pit-window suggestion")
if window["start_lap"] is None:
    st.write(window["status"])
else:
    st.write(f"{window['status']}: lap {window['start_lap']} to {window['end_lap']}" if window["end_lap"] else
             f"{window['status']}: lap {window['start_lap']} onward. The upper threshold was not reached in the evaluated range.")

st.subheader("Race holdout results by compound")
st.dataframe(result_rows, use_container_width=True, hide_index=True)
st.caption("Limitations: fixed fuel sensitivity, linear track trend, no dirty-air, engine-mode, ERS or tyre-temperature model. Practice-to-race extrapolation can fail when race tyre management differs.")
