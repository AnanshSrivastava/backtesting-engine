# src/backtest.py
import pandas as pd
import numpy as np

def run_backtest(df, initial_capital=100000):
    """
    Convert close-based crossover signals into positions and compute returns.

    A signal is only known after that session's close. This backtest assumes
    execution at the following session's close, so the position is shifted by
    two rows before it is applied to close-to-close returns: the signal row is
    not traded, and the return ending on the assumed fill row is not earned.
    Expects df to already have a 'signal' column from generate_signals().
    """
    # signal has: 1 (buy crossover), -1 (sell crossover), 0 (nothing happening)
    # position should be: 1 while holding, 0 while flat, carried forward between crossovers

    df['position'] = df['signal'].replace(0, np.nan)      # turn "nothing happening" into gaps
    df['position'] = df['position'].replace(-1, 0)    # sell = flat
    df['position'] = df['position'].ffill()             # carry forward last known state
    df['position'] = df['position'].fillna(0)         # handle rows before first crossover

    # Signal is confirmed at close t; assume fill at close t+1; earn returns
    # beginning with the close-to-close interval after that fill (row t+2).
    df['position'] = df['position'].shift(2).fillna(0)

    # gives day-over-day percentage change in adjusted close price
    df['daily_return'] = df['Adj Close'].pct_change()

    # Only earn the daily_return on days where position == 1
    df['strategy_return'] = df['position'] * df['daily_return']
    df['strategy_return'] = df['strategy_return'].fillna(0)  # fill NaN values with 0 for days without a position
    # Starting from initial_capital, compound the strategy returns day by day
    df['equity_curve'] = initial_capital * (1 + df['strategy_return']).cumprod()

    return df
