# app.py  (or streamlit_app.py on Streamlit Cloud)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# -----------------
# App title
# -----------------
st.set_page_config(page_title="Backtest App", layout="wide")
st.title("📈 Backtest App for Indian & Global Markets")

# -----------------
# Input fields
# -----------------
ticker = st.text_input("Ticker (e.g. ACC, RELIANCE, NIFTY, SENSEX)", value="ACC")
start_date = st.date_input("Start date", value=dt.date(2021, 9, 30))
end_date = st.date_input("End date", value=dt.date.today())
initial_cap = st.number_input("Initial capital", value=100000.0, format="%.2f")
pos_risk = st.slider("Per-trade Stop Loss (%)", 1, 10, 2) / 100.0
max_dd = st.slider("Max Drawdown (%)", 1, 20, 5) / 100.0
monthly_target = st.slider("Monthly Profit Target (%)", 1, 10, 5) / 100.0

run = st.button("🚀 Run Backtest")

# -----------------
# Helper: Normalize Indian tickers
# -----------------
def normalize_ticker(t):
    t_up = t.upper()
    if t_up in ("NIFTY", "NIFTY50"):
        return "^NSEI"
    if t_up in ("SENSEX", "BSE"):
        return "^BSESN"
    if t.startswith("^") or "." in t:
        return t
    return t_up + ".NS"

# -----------------
# Backtest function (simplified)
# -----------------
def run_backtest(ticker, start, end, initial_capital, pos_risk, max_dd, monthly_target):
    nt = normalize_ticker(ticker)
    st.info(f"Fetching data for **{nt}** from {start} to {end}")
    df = yf.download(nt, start=start, end=end, progress=False)

    if df.empty:
        st.error("No data found. Try different ticker.")
        return None, None

    # Use Adj Close if available, else Close
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Open" not in df.columns:
        st.error("No 'Open' prices available for this ticker.")
        return None, None

    # Example strategy: simple buy & hold to validate app (replace with your strategy)
    df["Equity"] = (df["Close"] / df["Close"].iloc[0]) * initial_capital

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Equity"], label="Equity")
    ax.set_title(f"Equity Curve: {nt}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    st.pyplot(fig)

    # Summary
    final_eq = df["Equity"].iloc[-1]
    total_return = (final_eq / initial_capital - 1) * 100
    st.success(f"Final Equity: {final_eq:.2f} | Total Return: {total_return:.2f}%")

    return df, final_eq

# -----------------
# Run backtest
# -----------------
if run:
    df, eq = run_backtest(ticker, start_date, end_date, initial_cap, pos_risk, max_dd, monthly_target)
    if df is not None:
        st.write("Sample Data:", df.head())
        st.download_button("Download Equity Data", df.to_csv().encode("utf-8"),
                           file_name="equity_curve.csv", mime="text/csv")
