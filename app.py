# streamlit_app.py
# Safe Streamlit backtester (drop-in replacement).
# This version avoids pkg_resources import (fixes ModuleNotFoundError)
# and surfaces exceptions in the UI so the app doesn't go blank.

import streamlit as st
import traceback
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io
import json

st.set_page_config(page_title="Backtest (debug-friendly)", layout="wide")
st.title("Backtest (debug-friendly) — Drop-in replacement")

# -------------------------
# Sidebar / quick diagnostics (no pkg_resources)
# -------------------------
st.sidebar.header("Quick Diagnostics")
if st.sidebar.button("Run import & env test"):
    try:
        with st.spinner("Testing imports and environment..."):
            st.sidebar.write(f"Python: {sys.version.split()[0]}")
            # show a small list of key modules and whether import succeeds
            modules = ["streamlit", "pandas", "numpy", "yfinance", "matplotlib"]
            mod_status = {}
            for m in modules:
                try:
                    __import__(m)
                    mod_status[m] = "ok"
                except Exception as e:
                    mod_status[m] = f"ERROR: {e}"
            st.sidebar.json(mod_status)
        st.sidebar.success("Environment test finished")
    except Exception as e:
        st.sidebar.error("Import test failed")
        st.sidebar.exception(e)

st.sidebar.write("---")
st.sidebar.write("If any modules show errors, add them to requirements.txt and redeploy.")

# -------------------------
# Main UI inputs
# -------------------------
col1, col2 = st.columns([2, 1])
with col1:
    ticker_input = st.text_input("Ticker (ACC / RELIANCE / NIFTY / SENSEX or ACC.NS / ^NSEI)", value="ACC")
    start_date = st.date_input("Start date", value=dt.date(2021, 9, 30))
    end_date = st.date_input("End date", value=dt.date.today())
    initial_capital = st.number_input("Initial capital", value=100000.0, step=1000.0, format="%.2f")
    max_hold_days = st.number_input("Max hold days", value=10, min_value=1, step=1)
with col2:
    pos_stop_pct = st.slider("Per-trade stop loss (%)", min_value=1.0, max_value=10.0, value=2.0) / 100.0
    overall_dd_pct = st.slider("Max overall drawdown (%)", min_value=1.0, max_value=20.0, value=5.0) / 100.0
    monthly_target_pct = st.slider("Monthly profit target (%)", min_value=1.0, max_value=20.0, value=5.0) / 100.0
    entry_threshold = st.number_input("Prior-day return entry threshold (fraction)", value=0.005, step=0.001, format="%.4f")

run_btn = st.button("Run Backtest")

st.write("Tip: use the sidebar *Import & env test* first if the app appears blank after deploy.")

# -------------------------
# Helpers
# -------------------------
def normalize_ticker(user_ticker: str) -> str:
    t = (user_ticker or "").strip()
    if not t:
        raise ValueError("Ticker cannot be empty.")
    tu = t.upper()
    if tu in ("NIFTY", "NIFTY50"):
        return "^NSEI"
    if tu in ("SENSEX", "BSE", "BSESN"):
        return "^BSESN"
    if tu.startswith("^") or "." in t:
        return t
    return tu + ".NS"

@st.cache_data(ttl=360_
