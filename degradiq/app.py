"""Unified DegradIQ Streamlit entry: one app, multiple pages.

Run from the repository root:

    streamlit run degradiq/app.py

Pages:
  * First-Stop Prediction — offline, engine-driven forecasts, replay and evaluation.
  * Tyre Degradation — the practice-vs-race degradation pipeline (needs FastF1 data).

Each page script also runs standalone (`streamlit run degradiq/dashboard_first_stop.py`).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

pages = [
    st.Page(str(HERE / "dashboard_first_stop.py"), title="First-Stop Prediction",
            icon="🏁", default=True),
    st.Page(str(HERE / "dashboard.py"), title="Tyre Degradation", icon="🛞"),
]

st.navigation(pages).run()
