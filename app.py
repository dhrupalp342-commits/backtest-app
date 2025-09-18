# streamlit_app.py
# Robust Streamlit backtester + debug wrapper.
# Replace your current app with this file.

import streamlit as st
import traceback
import sys
import pkg_resources
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io
import json

st.set_page_config(page_title="Backtest (debug + safe)", layout="wide")
st.title("Backtest (debug-friendly) — Drop-in replacement")

# -------------------------
# Sidebar / quick debug
# -------------------------
st.sidebar.header("Quick Diagnostics")
if st.sidebar.button("Run import & env test"):
    try:
        with st.spinner("Testing imports and environment..."):
            # show python version
            st.sidebar.write(f"Python: {sys.version.split()[0]}")
            pkgs = sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])
            st.sidebar.write("Top installed packages (first 40):")
            st.sidebar.write(pkgs[:40])
        st.sidebar.success("Imports OK")
    except Exception as e:
        st.sidebar.error("Import test failed")
        st.sidebar.exception(e)

st.sidebar.write("---")
st.sidebar.write("If you see import errors here, add missing packages to requirements.txt and redeploy.")

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

@st.cache_data(ttl=3600)
def download_data(ticker: str, start: str, end: str):
    # returns DataFrame or raises
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} in {start}..{end}")
    # prefer Adj Close
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Adj_Close"})
        price_col = "Adj_Close"
    else:
        price_col = "Close"
    if "Open" not in df.columns:
        raise RuntimeError("Downloaded data lacks 'Open' prices — cannot run intraday checks.")
    df = df[["Open", price_col]].rename(columns={price_col: "Close"})
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.cummax()
    dd = (peak - equity_series) / peak
    return float(dd.max())

# The core backtest (same protected logic as earlier, but returns errors as strings)
def run_backtest_core(df: pd.DataFrame,
                      initial_capital: float,
                      pos_stop_pct: float,
                      overall_dd_pct: float,
                      monthly_target_pct: float,
                      max_hold_days: int,
                      entry_threshold: float):
    try:
        cash = initial_capital
        positions = []
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

            if today_month not in monthly_realized:
                monthly_realized[today_month] = 0.0
                month_start_equity[today_month] = cash + sum([p["shares"] * today_close for p in positions])

            realized_today = 0.0
            intraday_low = min(today_open, today_close)
            remaining_positions = []
            for pos in positions:
                exit_price = None
                reason = None
                if intraday_low <= pos["stop_price"]:
                    exit_price = pos["stop_price"]
                    reason = "stop"
                else:
                    hold_days = (today_date - pos["entry_date"]).days
                    if hold_days >= max_hold_days:
                        exit_price = today_close
                        reason = "time_exit"
                if exit_price is not None:
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    realized_today += pnl
                    cash += pos["shares"] * exit_price
                    trades.append({
                        "entry_date": pos["entry_date"].isoformat(),
                        "exit_date": today_date.isoformat(),
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "reason": reason
                    })
                else:
                    remaining_positions.append(pos)
            positions = remaining_positions
            monthly_realized[today_month] += realized_today

            equity = cash + sum([p["shares"] * today_close for p in positions])
            eq_hist = [pt["equity"] for pt in equity_points] + [equity]
            running_dd = compute_max_drawdown(pd.Series(eq_hist)) if eq_hist else 0.0

            paused_for_month = monthly_realized[today_month] >= (month_start_equity[today_month] * monthly_target_pct)

            signal = False
            pr = row["prior_ret"]
            if (not paused_for_month) and (pd.notna(pr) and pr > entry_threshold):
                signal = True

            if signal:
                risk_amount = equity * pos_stop_pct
                stop_price = today_open * (1.0 - pos_stop_pct)
                stop_distance = max(1e-6, today_open - stop_price)
                shares = int(risk_amount // stop_distance)
                if shares > 0 and shares * today_open <= cash:
                    # simulate worst-case if stop triggered immediately
                    worst_equity_if_stop = cash - (shares * today_open) + (shares * stop_price) + sum([p["shares"] * today_close for p in positions])
                    allowed_min_equity = initial_capital * (1.0 - overall_dd_pct)
                    if worst_equity_if_stop < allowed_min_equity:
                        trades.append({
                            "entry_date": today_date.isoformat(),
                            "exit_date": None,
                            "entry_price": today_open,
                            "exit_price": None,
                            "shares": 0,
                            "pnl": 0.0,
                            "reason": "entry_skipped_would_breach_overall_drawdown"
                        })
                    else:
                        cash -= shares * today_open
                        positions.append({
                            "entry_date": today_date,
                            "entry_price": today_open,
                            "shares": shares,
                            "stop_price": stop_price
                        })

            # check worst-case intraday across all positions
            worst_equity = cash + sum([
                (pos["stop_price"] if pos["stop_price"] < intraday_low else today_close) * pos["shares"]
                for pos in positions
            ])
            allowed_min_equity = initial_capital * (1.0 - overall_dd_pct)
            if worst_equity < allowed_min_equity:
                # emergency close
                for pos in positions:
                    exit_price = today_close
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    cash += pos["shares"] * exit_price
                    trades.append({
                        "entry_date": pos["entry_date"].isoformat(),
                        "exit_date": today_date.isoformat(),
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "reason": "emergency_close_to_protect_overall_drawdown"
                    })
                positions = []
                equity = cash
                if equity < allowed_min_equity:
                    shutdown_info["happened"] = True
                    shutdown_info["date"] = today_date.isoformat()
                    shutdown_info["reason"] = "overall_drawdown_exceeded_after_emergency_close"
                    equity_points.append({"date": idx.isoformat(), "equity": float(equity)})
                    break

            equity = cash + sum([p["shares"] * today_close for p in positions])
            equity_points.append({"date": idx.isoformat(), "equity": float(equity)})

        equity_df = pd.DataFrame(equity_points)
        trades_df = pd.DataFrame(trades)

        if not equity_df.empty:
            final_equity = float(equity_df["equity"].iloc[-1])
            max_dd = compute_max_drawdown(equity_df["equity"])
            max_profit = float((pd.Series(equity_df["equity"]).cummax() - initial_capital).max())
        else:
            final_equity = initial_capital
            max_dd = 0.0
            max_profit = 0.0

        report = {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": (final_equity / initial_capital - 1.0) * 100.0,
            "max_drawdown_pct": max_dd,
            "max_profit_abs": max_profit,
            "trades_count": len(trades_df),
            "shutdown": shutdown_info
        }

        return {"ok": True, "report": report, "equity_df": equity_df, "trades_df": trades_df}
    except Exception as e:
        tb = traceback.format_exc()
        return {"ok": False, "error": str(e), "traceback": tb}

# -------------------------
# Run & display
# -------------------------
if run_btn:
    try:
        norm_ticker = normalize_ticker(ticker_input)
    except Exception as e:
        st.error(f"Ticker normalization error: {e}")
        st.exception(e)
        st.stop()

    st.write(f"Normalized ticker: **{norm_ticker}**")
    try:
        df = download_data(norm_ticker, start_date.isoformat(), end_date.isoformat())
        st.success(f"Data downloaded: {len(df)} rows")
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        st.write("Full traceback (for debugging):")
        st.text(traceback.format_exc())
        st.stop()

    with st.spinner("Running backtest (safe mode)..."):
        res = run_backtest_core(df,
                                initial_capital=float(initial_capital),
                                pos_stop_pct=float(pos_stop_pct),
                                overall_dd_pct=float(overall_dd_pct),
                                monthly_target_pct=float(monthly_target_pct),
                                max_hold_days=int(max_hold_days),
                                entry_threshold=float(entry_threshold))
    if not res["ok"]:
        st.error("Backtest failed with an exception. See details below.")
        st.text(res.get("error", ""))
        st.text(res.get("traceback", ""))
        st.stop()

    report = res["report"]
    equity_df = res["equity_df"]
    trades_df = res["trades_df"]

    # Summary
    st.subheader("Summary")
    st.metric("Initial capital", f"{report['initial_capital']:,.2f}")
    st.metric("Final equity", f"{report['final_equity']:,.2f}", delta=f"{report['total_return_pct']:.2f}%")
    st.write(f"Max drawdown observed: {report['max_drawdown_pct']:.2%}")
    st.write(f"Max absolute profit above start: {report['max_profit_abs']:.2f}")
    if report["shutdown"]["happened"]:
        st.warning(f"Shutdown occurred on {report['shutdown']['date']}: {report['shutdown']['reason']}")

    # Equity plot
    if not equity_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"], label="Equity")
        ax.axhline(report['initial_capital'] * (1.0 - overall_dd_pct), color="red", linestyle="--", label="Allowed min equity")
        ax.set_title("Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.legend()
        st.pyplot(fig)

    # Trades & downloads
    st.subheader("Trades")
    if trades_df.empty:
        st.write("No trades executed.")
    else:
        st.dataframe(trades_df)
        buf1 = io.StringIO()
        equity_df.to_csv(buf1, index=False)
        st.download_button("Download equity_curve.csv", buf1.getvalue().encode("utf-8"), file_name="equity_curve.csv")
        buf2 = io.StringIO()
        trades_df.to_csv(buf2, index=False)
        st.download_button("Download trades.csv", buf2.getvalue().encode("utf-8"), file_name="trades.csv")

    st.subheader("Report JSON")
    st.json(report)

else:
    st.info("Set parameters and press 'Run Backtest'. Use the sidebar import/env test if you get a blank page.")

