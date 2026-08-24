# src/backtest.py
import pandas as pd
import numpy as np

def run_backtest(df, initial_capital=100000):
    """
    Convert crossover signals into a continuous position, apply lookahead-safe
    shifting, and compute strategy returns and equity curve.
    Expects df to already have a 'signal' column from generate_signals().
    """
    # signal has: 1 (buy crossover), -1 (sell crossover), 0 (nothing happening)
    # position should be: 1 while holding, 0 while flat, carried forward between crossovers

    df['position'] = df['signal'].replace(0, np.nan)      # turn "nothing happening" into gaps
    df['position'] = df['position'].replace(-1, 0)    # sell = flat
    df['position'] = df['position'].ffill()             # carry forward last known state
    df['position'] = df['position'].fillna(0)         # handle rows before first crossover

    # We can only ACT on it starting day N+1.
    df['position'] = df['position'].shift(1)  # shift down by 1 to avoid lookahead bias

    # gives day-over-day percentage change in adjusted close price
    df['daily_return'] = df['Adj Close'].pct_change()

    # Only earn the daily_return on days where position == 1
    df['strategy_return'] = df['position'] * df['daily_return']
    df['strategy_return'] = df['strategy_return'].fillna(0)  # fill NaN values with 0 for days without a position
    # Starting from initial_capital, compound the strategy returns day by day
    df['equity_curve'] = initial_capital * (1 + df['strategy_return']).cumprod()

    return df