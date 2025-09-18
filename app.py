# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Simple Backtest", layout="wide")

st.title("Simple SMA Crossover Backtest (Free)")

# UI inputs
symbol = st.text_input("Ticker (Yahoo)", value="AAPL")
start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365*2))
end_date = st.date_input("End date", value=datetime.now().date())
short_window = st.number_input("Short SMA window", min_value=1, value=20)
long_window = st.number_input("Long SMA window", min_value=1, value=50)
initial_cash = st.number_input("Initial cash (INR / USD)", min_value=100.0, value=10000.0, step=100.0)
run_bt = st.button("Run Backtest")

# Simple strategy using backtrader
class SmaCross(bt.Strategy):
    params = dict(
        pfast=20,
        pslow=50,
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)
        sma2 = bt.ind.SMA(period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.position and self.crossover > 0:
            # Buy with all cash
            size = int(self.broker.getcash() / self.data.close[0])
            if size > 0:
                self.buy(size=size)
        elif self.position and self.crossover < 0:
            self.close()

def run_backtest(df, pfast, pslow, cash):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, pfast=pfast, pslow=pslow)
    # record starting value
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    pnl = end_value - start_value
    strat = results[0]
    # Get analyzers or create simple equity curve by running cerebro.plot? We'll generate equity using vector of trades:
    return cerebro, strat, start_value, end_value, pnl

if run_bt:
    st.info(f"Fetching {symbol} data from Yahoo Finance...")
    try:
        raw = yf.download(symbol, start=start_date, end=end_date)
        if raw.empty:
            st.error("No data returned for that symbol / date range.")
        else:
            df = raw[['Open','High','Low','Close','Adj Close','Volume']].copy()
            df.rename(columns={'Adj Close':'AdjClose'}, inplace=True)
            st.success(f"Downloaded {len(df)} rows.")
            with st.spinner("Running backtest..."):
                cerebro, strat, start_value, end_value, pnl = run_backtest(df,
                                                                           pfast=int(short_window),
                                                                           pslow=int(long_window),
                                                                           cash=float(initial_cash))
            st.write("### Summary")
            st.write(f"Start portfolio value: {start_value:,.2f}")
            st.write(f"End portfolio value: {end_value:,.2f}")
            st.write(f"P/L: {pnl:,.2f}")

            st.write("### Equity curve")
            # produce equity curve by re-running with observer to collect values step by step
            class ValueObserver(bt.Analyzer):
                def start(self):
                    self.values = []
                def next(self):
                    self.values.append(self.strategy.broker.getvalue())

            cerebro2 = bt.Cerebro()
            cerebro2.broker.setcash(float(initial_cash))
            cerebro2.adddata(bt.feeds.PandasData(dataname=df))
            cerebro2.addstrategy(SmaCross, pfast=int(short_window), pslow=int(long_window))
            cerebro2.addanalyzer(ValueObserver, _name="val")
            res2 = cerebro2.run()
            eq = res2[0].analyzers.val.values

            eq_df = pd.DataFrame({"equity": eq})
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(eq_df.index, eq_df['equity'])
            ax.set_title(f"Equity Curve: {symbol}")
            ax.set_xlabel("Bars")
            ax.set_ylabel("Portfolio Value")
            st.pyplot(fig)

            st.write("### Basic trades (last 10 trades from strategy broker)")
            # backtrader doesn't expose trades easily without analyzers; show trade log if any (not implemented here)
            st.info("If you want full trade logs, add analyzers (e.g., Trades) or persist orders in the strategy code.")

    except Exception as e:
        st.error(f"Error: {e}")
