# streamlit_app.py
# Full Streamlit backtester: per-trade stop, overall max-drawdown safety, monthly target, and reporting.
# Replace your current app with this file.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="Backtest (safe drawdown)", layout="wide")
st.title("Backtest (enforces per-trade stop & overall drawdown)")

# -------------------------
# UI inputs
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input("Ticker (ACC / RELIANCE / NIFTY / SENSEX or ACC.NS / ^NSEI)", value="ACC")
    start_date = st.date_input("Start date", value=dt.date(2021, 9, 30))
    end_date = st.date_input("End date", value=dt.date.today())
    initial_capital = st.number_input("Initial capital", value=100000.0, step=1000.0, format="%.2f")
with col2:
    pos_stop_pct = st.slider("Per-trade stop loss (%)", min_value=1.0, max_value=10.0, value=2.0) / 100.0
    overall_dd_pct = st.slider("Max overall drawdown (%)", min_value=1.0, max_value=20.0, value=5.0) / 100.0
    monthly_target_pct = st.slider("Monthly profit target (%)", min_value=1.0, max_value=20.0, value=5.0) / 100.0
    max_hold_days = st.number_input("Max hold days", value=10, min_value=1, step=1)

run_btn = st.button("Run Backtest")

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

def fetch_df(ticker, start, end):
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in range {start} to {end}.")
    # pick Adj Close if present, else Close
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Adj_Close"})
        price_col = "Adj_Close"
    else:
        price_col = "Close"
    if "Open" not in df.columns:
        raise RuntimeError("Downloaded data has no 'Open' column.")
    df = df[["Open", price_col]].rename(columns={price_col: "Close"})
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(equity_series: pd.Series) -> float:
    peak = equity_series.cummax()
    dd = (peak - equity_series) / peak
    return float(dd.max()) if len(dd) else 0.0

# -------------------------
# Backtest logic
# -------------------------
def backtest(df: pd.DataFrame,
             initial_capital: float,
             pos_stop_pct: float,
             overall_dd_pct: float,
             monthly_target_pct: float,
             max_hold_days: int):
    # We'll use a simple momentum entry: prior day return > 0.5%
    entry_threshold = 0.005

    cash = initial_capital
    positions = []  # list of dicts: entry_date, entry_price, shares, stop_price
    equity_points = []
    trades = []
    monthly_realized = {}
    month_start_equity = {}
    shutdown_info = {"happened": False, "date": None, "reason": None}

    df2 = df.copy()
    df2["prior_ret"] = df2["Close"].pct_change().shift(1)
    df2["month"] = df2.index.to_period("M")

    for idx, row in df2.iterrows():
        today_open = float(row["Open"])
        today_close = float(row["Close"])
        today_date = idx.date()
        today_month = row["month"]

        # init month bookkeeping
        if today_month not in monthly_realized:
            monthly_realized[today_month] = 0.0
            month_start_equity[today_month] = cash + sum([p["shares"] * today_close for p in positions])

        realized_today = 0.0
        # 1) check existing positions for stop or time exit (use intraday low approx = min(open, close))
        intraday_low = min(today_open, today_close)
        remaining_position_

