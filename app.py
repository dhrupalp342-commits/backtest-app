#!/usr/bin/env python3
"""
backtest_india.py

Like the prior backtester but with nicer handling for Indian market tickers.

Usage:
  python backtest_india.py --ticker ACC --start 2021-09-30 --end 2025-09-18 --initial 100000
  python backtest_india.py --ticker NIFTY --start 2021-09-30 --end 2025-09-18

Notes:
 - For NSE stocks: use plain name (ACC) or full (ACC.NS). The script will try to normalize.
 - For indices: use NIFTY / SENSEX (case-insensitive) or pass Yahoo tickers like ^NSEI / ^BSESN directly.
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import json
import matplotlib.pyplot as plt
import sys

# -------------------------
# Defaults
# -------------------------
DEFAULTS = {
    'position_risk_pct': 0.02,   # per-trade stop (2%)
    'max_drawdown_pct': 0.05,    # overall drawdown allowed (5%)
    'monthly_target_pct': 0.05,  # monthly profit target (5%)
    'max_hold_days': 10,         # exit after this many days
    'entry_threshold': 0.005,    # prior day return > 0.5% -> entry
    'shares_per_trade': None
}

# -------------------------
# Helper: Normalize user input for Indian markets & general
# -------------------------
def normalize_ticker(user_ticker: str) -> str:
    """
    Convert common user inputs into Yahoo Finance tickers:
     - 'ACC' -> 'ACC.NS'
     - 'reliance' -> 'RELIANCE.NS'
     - 'NIFTY' -> '^NSEI'
     - 'SENSEX' -> '^BSESN'
     - If user passes an explicit suffix (e.g. '.NS', '^'), keep it.
    """
    if not user_ticker or not user_ticker.strip():
        raise ValueError("Empty ticker")
    t = user_ticker.strip()
    t_up = t.upper()

    # index shortcuts
    if t_up in ('NIFTY', 'NIFTY50'):
        return '^NSEI'
    if t_up in ('SENSEX', 'BSE', 'BSESN'):
        return '^BSESN'

    # If it already looks like a Yahoo ticker (starts with ^ for index or has dot suffix),
    # return as-is
    if t.startswith('^') or '.' in t:
        return t

    # Common pattern: assume NSE equity -> append .NS
    return t.upper() + '.NS'

# -------------------------
# Data fetch & helpers
# -------------------------
def fetch_price_series(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}. Check symbol & internet connection.")
    # Prefer 'Adj Close', fallback to 'Close'
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Adj_Close'})
        price_col = 'Adj_Close'
    elif 'Close' in df.columns:
        df = df.rename(columns={'Close': 'Adj_Close'})
        price_col = 'Adj_Close'
    else:
        raise ValueError("No Close or Adj Close price found in downloaded data.")
    # Index data (e.g. ^NSEI) usually includes Open & Close; ensure Open exists
    if 'Open' not in df.columns:
        raise ValueError("Downloaded data has no 'Open' column; intraday logic won't work.")
    df = df[['Open', price_col]].rename(columns={price_col: 'Close'})
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(equity_series):
    peak = equity_series.cummax()
    drawdown = (peak - equity_series) / peak
    return drawdown.max()

# -------------------------
# Backtest engine & report (same design as prior)
# -------------------------
def run_backtest_and_report(df, params):
    initial_capital = float(params['initial_capital'])
    position_risk_pct = float(params['position_risk_pct'])
    max_drawdown_pct = float(params['max_drawdown_pct'])
    monthly_target_pct = float(params['monthly_target_pct'])
    entry_threshold = float(params['entry_threshold'])
    max_hold_days = int(params['max_hold_days'])
    shares_per_trade = params.get('shares_per_trade', None)

    cash = initial_capital
    positions = []
    equity_curve = []
    trades = []
    monthly_realized = {}
    month_start_equity = {}
    shutdown = {'happened': False, 'date': None, 'reason': None}
    stop_hits = 0

    df = df.copy()
    df['prior_ret'] = df['Close'].pct_change().shift(1)
    df['month'] = df.index.to_period('M')

    for idx, row in df.iterrows():
        today_open = row['Open']
        today_close = row['Close']
        today_date = idx.date()
        today_month = row['month']

        if today_month not in monthly_realized:
            monthly_realized[today_month] = 0.0
            month_start_equity[today_month] = cash + sum([p['shares'] * today_close for p in positions])

        realized_pnl_today = 0.0
        remaining_positions = []
        intraday_low = min(today_open, today_close)
        for pos in positions:
            exit_price = None
            exit_reason = None
            if intraday_low <= pos['stop_price']:
                exit_price = pos['stop_price']
                exit_reason = 'stop'
                stop_hits += 1
            else:
                hold_days = (today_date - pos['entry_date']).days
                if hold_days >= max_hold_days:
                    exit_price = today_close
                    exit_reason = 'time_exit'
            if exit_price is not None:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                realized_pnl_today += pnl
                cash += pos['shares'] * exit_price
                trades.append({
                    'entry_date': pos['entry_date'].isoformat(),
                    'exit_date': today_date.isoformat(),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'reason': exit_reason
                })
            else:
                remaining_positions.append(pos)
        positions = remaining_positions

        monthly_realized[today_month] += realized_pnl_today

        equity = cash + sum([p['shares'] * today_close for p in positions])

        paused_for_month = False
        start_eq = month_start_equity.get(today_month, initial_capital)
        if monthly_realized[today_month] >= start_eq * monthly_target_pct:
            paused_for_month = True

        signal = False
        pr = row['prior_ret']
        if (not paused_for_month) and pd.notna(pr) and pr > entry_threshold:
            signal = True

        if signal:
            equity = cash + sum([p['shares'] * today_close for p in positions])
            risk_amount = equity * position_risk_pct
            stop_price = today_open * (1.0 - position_risk_pct)
            stop_distance = max(1e-6, (today_open - stop_price))
            if shares_per_trade:
                shares = int(shares_per_trade)
            else:
                shares = int(max(0, risk_amount // stop_distance))
            if shares > 0 and shares * today_open <= cash:
                cash -= shares * today_open
                positions.append({
                    'entry_date': today_date,
                    'entry_price': today_open,
                    'shares': shares,
                    'stop_price': stop_price
                })

        equity = cash + sum([p['shares'] * today_close for p in positions])
        equity_curve.append({'date': idx.isoformat(), 'equity': float(equity), 'cash': float(cash), 'open_positions': len(positions),
                             'monthly_realized': float(monthly_realized[today_month])})

        eq_series = pd.Series([e['equity'] for e in equity_curve])
        max_dd = compute_max_drawdown(eq_series)
        if max_dd >= max_drawdown_pct:
            for pos in positions:
                exit_price = today_close
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cash += pos['shares'] * exit_price
                trades.append({
                    'entry_date': pos['entry_date'].isoformat(),
                    'exit_date': today_date.isoformat(),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'reason': 'max_drawdown_shutdown'
                })
            positions = []
            equity = cash
            equity_curve[-1]['equity'] = float(equity)
            shutdown = {'happened': True, 'date': today_date.isoformat(), 'reason': 'max_drawdown_reached'}
            break

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    final_equity = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else float(initial_capital)
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0
    max_dd = float(compute_max_drawdown(pd.Series(equity_df['equity']))) if not equity_df.empty else 0.0
    trades_count = len(trades_df)

    # monthly returns
    if not equity_df.empty:
        eq_temp = equity_df.copy()
        eq_temp['date'] = pd.to_datetime(eq_temp['date'])
        eq_temp = eq_temp.set_index('date')
        eq_temp['month'] = eq_temp.index.to_period('M')
        month_start = eq_temp.groupby('month')['equity'].first()
        month_end = eq_temp.groupby('month')['equity'].last()
        monthly_returns = ((month_end / month_start - 1.0) * 100.0).dropna().to_dict()
        monthly_returns = {str(k): float(v) for k, v in monthly_returns.items()}
    else:
        monthly_returns = {}

    report = {
        'meta': {
            'ticker': params['ticker'],
            'normalized_ticker': params['normalized_ticker'],
            'start': params['start'],
            'end': params['end'],
            'initial_capital': initial_capital,
            'params': {
                'position_risk_pct': position_risk_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'monthly_target_pct': monthly_target_pct,
                'entry_threshold': entry_threshold,
                'max_hold_days': max_hold_days,
                'shares_per_trade': shares_per_trade
            }
        },
        'summary': {
            'final_equity': final_equity,
            'total_return_pct': round(total_return_pct, 4),
            'max_drawdown_pct': round(max_dd, 6),
            'trades_count': trades_count,
            'stophits_count': stop_hits
        },
        'monthly_returns_pct': monthly_returns,
        'trades': trades,
        'equity_curve': equity_curve,
        'constraints': {
            'per_trade_stop_specified_pct': position_risk_pct,
            'max_drawdown_specified_pct': max_drawdown_pct,
            'monthly_target_specified_pct': monthly_target_pct,
            'shutdown': shutdown
        }
    }

    return report, equity_df, trades_df

# -------------------------
# CLI and main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Backtest supporting Indian markets and friendly ticker input.")
    p.add_argument('--ticker', type=str, help='Ticker (e.g. ACC, ACC.NS, RELIANCE, ^NSEI, NIFTY)')
    p.add_argument('--start', type=str, default='2021-09-30', help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=dt.date.today().isoformat(), help='End date YYYY-MM-DD')
    p.add_argument('--initial', type=float, default=100000.0, help='Initial capital')
    p.add_argument('--pos_risk', type=float, default=DEFAULTS['position_risk_pct'], help='Per-trade risk fraction (e.g. 0.02)')
    p.add_argument('--max_dd', type=float, default=DEFAULTS['max_drawdown_pct'], help='Max drawdown fraction (e.g. 0.05)')
    p.add_argument('--monthly_target', type=float, default=DEFAULTS['monthly_target_pct'], help='Monthly profit target fraction (e.g. 0.05)')
    p.add_argument('--entry_thresh', type=float, default=DEFAULTS['entry_threshold'], help='Entry threshold (prior day return)')
    p.add_argument('--max_hold', type=int, default=DEFAULTS['max_hold_days'], help='Max hold days')
    p.add_argument('--shares', type=int, default=None, help='Fixed shares per trade (override risk sizing)')
    return p.parse_args()

def main():
    args = parse_args()
    ticker = args.ticker
    if not ticker:
        ticker = input("Enter ticker (e.g. ACC, NIFTY): ").strip()
        if not ticker:
            print("Ticker required. Exiting.")
            sys.exit(1)

    try:
        normalized = normalize_ticker(ticker)
    except Exception as e:
        print("Ticker normalization error:", e)
        sys.exit(1)

    params = {
        'ticker': ticker,
        'normalized_ticker': normalized,
        'start': args.start,
        'end': args.end,
        'initial_capital': args.initial,
        'position_risk_pct': args.pos_risk,
        'max_drawdown_pct': args.max_dd,
        'monthly_target_pct': args.monthly_target,
        'entry_threshold': args.entry_thresh,
        'max_hold_days': args.max_hold,
        'shares_per_trade': args.shares
    }

    print(f"Using normalized ticker: {normalized}  (from input: {ticker})")
    print(f"Fetching {normalized} from {args.start} to {args.end} ...")
    try:
        df = fetch_price_series(normalized, args.start, args.end)
    except Exception as e:
        print("Data fetch error:", e)
        sys.exit(1)
    print(f"Data rows: {len(df)}. Running backtest...")

    report, equity_df, trades_df = run_backtest_and_report(df, params)

    # Save outputs
    equity_df.to_csv("equity_curve.csv", index=False)
    trades_df.to_csv("trades.csv", index=False)
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n=== Backtest Report Summary ===")
    meta = report['meta']
    summary = report['summary']
    constraints = report['constraints']
    print(f"Original input: {meta['ticker']}")
    print(f"Normalized ticker used: {meta['normalized_ticker']}")
    print(f"Period: {meta['start']} -> {meta['end']}")
    print(f"Initial capital: {meta['initial_capital']}")
    print(f"Final equity: {summary['final_equity']:.2f}")
    print(f"Total return: {summary['total_return_pct']:.2f}%")
    print(f"Max drawdown observed: {summary['max_drawdown_pct']:.2%}")
    print(f"Trades: {summary['trades_count']}  Stop hits: {summary['stophits_count']}")
    if constraints['shutdown']['happened']:
        print(f"Shutdown occurred on {constraints['shutdown']['date']}: {constraints['shutdown']['reason']}")
    else:
        print("No max-drawdown shutdown.")
    print("\nMonthly returns (last 12):")
    mr = report['monthly_returns_pct']
    last12 = dict(list(mr.items())[-12:])
    for m, r in last12.items():
        print(f"  {m}: {r:.2f}%")

    print("\nSaved: equity_curve.csv, trades.csv, report.json")

    # Plot equity curve (if possible)
    if not equity_df.empty:
        plt.figure(figsize=(10,5))
        plt.plot(pd.to_datetime(equity_df['date']), equity_df['equity'])
        plt.title(f"Equity curve: {meta['normalized_ticker']}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

