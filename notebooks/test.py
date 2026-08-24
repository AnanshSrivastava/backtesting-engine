# notebooks/test.py
"""
End-to-end pipeline test: fetch data, generate signals, run backtest, compute metrics.
"""
import sys
import os

"adding the parent directory to sys.path to import fetch_data from src"

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fetch_data import fetch_data
from src.strategy import generate_signals
from src.backtest import run_backtest
from src.metrics import sharpe_ratio, cagr, max_drawdown

TICKER = "RELIANCE.NS"
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

df = fetch_data(TICKER, START_DATE, END_DATE)
benchmark_df = fetch_data(BENCHMARK_TICKER, START_DATE, END_DATE)
benchmark_df["daily_return"] = benchmark_df["Adj Close"].pct_change()

if df is None:
    print(f"No data for {TICKER}")
else:
    df_signals = generate_signals(df.copy(), verbose=False)
    df_signals = df_signals.dropna(subset=['SMA_long'])
    backtest_results = run_backtest(df_signals)

    strategy_cagr = cagr(backtest_results['equity_curve'])
    sharpe = sharpe_ratio(backtest_results['strategy_return'], benchmark_df['daily_return'])
    strategy_max_dd = max_drawdown(backtest_results['equity_curve'])
    buy_hold_max_dd = max_drawdown(backtest_results['Adj Close'])

    buy_hold_return = (backtest_results['Adj Close'].iloc[-1] / backtest_results['Adj Close'].iloc[0]) - 1
    strategy_total_return = (backtest_results['equity_curve'].iloc[-1] / backtest_results['equity_curve'].iloc[0]) - 1
    nifty_return = (benchmark_df['Adj Close'].iloc[-1] / benchmark_df['Adj Close'].iloc[0]) - 1

    print(f"--- {TICKER} | {START_DATE} to {END_DATE} ---")
    print(f"Strategy CAGR:            {strategy_cagr:.2%}")
    print(f"Sharpe Ratio:              {sharpe:.4f}")
    print(f"Strategy Max Drawdown:     {strategy_max_dd:.2%}")
    print(f"Buy & Hold Max Drawdown:   {buy_hold_max_dd:.2%}")
    print(f"Strategy Total Return:     {strategy_total_return:.2%}")
    print(f"Buy & Hold Total Return:   {buy_hold_return:.2%}")
    print(f"Nifty 50 Total Return:     {nifty_return:.2%}")