"""
============================================================
Drone Route Planner — Jaipur Airspace
============================================================
Main entry point for Hugging Face Spaces deployment.
Lets users choose between Single Drone or Fleet Mode.
============================================================
"""

import os
import sys

# Add src directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

import streamlit as st

st.set_page_config(
    page_title="Drone Route Planner - Jaipur",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mode Selector ──
mode = st.sidebar.radio(
    "🛸 Navigation Mode",
    ["🚁 Single Drone", "🚁🚁 Multi-Drone Fleet"],
    index=0,
)

if mode == "🚁 Single Drone":
    from step6_dashboard import main
    main()
else:
    from step8_fleet_dashboard import main
    main()
