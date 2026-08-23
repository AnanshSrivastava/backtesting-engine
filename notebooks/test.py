# notebooks/test.py
import sys
import os
"adding the parent directory to sys.path to import fetch_data from src"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fetch_data import fetch_data
from src.strategy import generate_signals
from src.backtest import run_backtest

df = fetch_data("RELIANCE.NS", "2020-01-01", "2024-01-01")
if df is not None:
    print("--- verbose=False ---")
    df_signals = generate_signals(df.copy(), verbose=False) #calling the generate_signals function to get the signals for the strategy
    df_signals = df_signals.dropna(subset=['SMA_long'])  # Drop rows where SMA values are NaN
    backtest_results = run_backtest(df_signals) #calling the run_backtest function to get the backtest results

    print("Backtest Results:")
    print(backtest_results[['Adj Close', 'SMA_short', 'SMA_long', 'signal', 'position', 'daily_return', 'strategy_return', 'equity_curve']].tail(50))
else:
    print("No data for RELIANCE.NS")