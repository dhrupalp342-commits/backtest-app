# streamlit_app.py
import streamlit as st
import traceback
import sys
import time

st.set_page_config(page_title="Backtest App (debug)", layout="wide")

st.title("Backtest App — Debug / Safe Runner")

# show python version and installed libs (small)
st.sidebar.header("Environment")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
if st.sidebar.button("Show installed packages (top 30)"):
    import pkg_resources
    pkgs = sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])[:30]
    st.sidebar.write(pkgs)

# Provide a quick sanity test
st.header("Sanity checks")
st.write("Press the button below to run a quick import + minimal test. This avoids running heavy downloads during import.")

if st.button("Run import & minimal test"):
    try:
        with st.spinner("Running import test..."):
            # lightweight imports here to test if requirements installed
            import pandas as pd
            import numpy as np
            import yfinance as yf
            import matplotlib.pyplot as plt
            import datetime as dt
            time.sleep(0.2)
        st.success("Basic imports succeeded.")
    except Exception as e:
        st.error("Import failed — check requirements.txt and logs.")
        st.exception(e)
        st.stop()

st.write("---")

# Main backtest runner (only runs on demand)
st.header("Run Backtest")
ticker_input = st.text_input("Ticker (e.g. ACC, ACC.NS, NIFTY)", value="ACC")
start_date = st.date_input("Start date", value=dt.date(2021,9,30))
end_date = st.date_input("End date", value=dt.date.today())
initial_cap = st.number_input("Initial capital", value=100000.0, format="%.2f")
pos_risk = st.number_input("Per-trade stop %", value=2.0, min_value=0.1, max_value=50.0) / 100.0
max_dd = st.number_input("Max drawdown %", value=5.0, min_value=0.1, max_value=100.0) / 100.0

run = st.button("Run Backtest Now")

# Keep heavy code inside function and wrapped by try/except
def safe_run_backtest(ticker, start, end, initial_capital, pos_risk, max_dd):
    try:
        # Lazy import of heavy modules
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime

        # Normalize ticker for India quick helper
        def normalize_ticker(t):
            if not t:
                raise ValueError("Empty ticker")
            up = t.upper()
            if up in ("NIFTY","NIFTY50"):
                return "^NSEI"
            if up in ("SENSEX","BSE"):
                return "^BSESN"
            if up.startswith("^") or "." in t:
                return t
            return up + ".NS"

        nt = normalize_ticker(ticker)
        st.write(f"Using ticker: **{nt}**")

        with st.spinner("Downloading data (yfinance)..."):
            df = yf.download(nt, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df.empty:
            st.error("No data returned by yfinance for this ticker & date range.")
            return

        # Select adj close fallback
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Adj_Close'})
            price_col = 'Adj_Close'
        else:
            price_col = 'Close'
        if 'Open' not in df.columns:
            st.error("Downloaded data does not contain 'Open' column — cannot run this intraday-ish backtest.")
            return

        df = df[['Open', price_col]].rename(columns={price_col: 'Close'})
        st.write(f"Data loaded: {len(df)} rows. Showing head:")
        st.dataframe(df.head())

        # simple sample equity curve so we validate plotting works
        equity_series = (df['Close'] / df['Close'].iloc[0]) * initial_capital
        fig, ax = plt.subplots()
        ax.plot(equity_series.index, equity_series.values)
        ax.set_title("Sample equity (price normalized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        st.pyplot(fig)

        st.success("Minimal run finished — now plug in your backtest function here.")
        return {"status": "ok", "rows": len(df)}
    except Exception as e:
        st.error("Backtest errored. See details below.")
        st.exception(e)
        st.write("Traceback (for logs):")
        st.text(traceback.format_exc())
        # also print to logs for Streamlit Cloud
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}

if run:
    result = safe_run_backtest(ticker_input, start_date, end_date, initial_cap, pos_risk, max_dd)
    st.write("Result:", result)
