import numpy as np
import pandas as pd


def run_backtest(df, initial_capital=100000, execution_lag=2, cost_per_side=0.0):
    """
    Turn crossover signals into positions, returns and an equity curve.

    Expects df to have 'signal' (from generate_signals) and 'Adj Close'.

    Execution timing: a signal is confirmed at the close of day T. With
    execution_lag=2 the trade happens at the close of day T+1 and the first
    return earned is T+1 to T+2. With execution_lag=1 the trade happens at
    the signal-day close itself (less conservative). Lags below 1 would
    trade on information not yet available, so they are rejected.

    Costs: cost_per_side is a fraction of traded value (0.001 = 10 bps),
    charged on every change of position. A full round trip pays it twice.
    The cost is deducted on the first day the new position earns a return.

    Adds: position, daily_return, turnover, cost, strategy_return,
    equity_curve.
    """
    if execution_lag < 1:
        raise ValueError("execution_lag must be at least 1 (lag 0 would use same-bar information)")
    if cost_per_side < 0:
        raise ValueError("cost_per_side must be non-negative")

    # 1 while holding, 0 while flat, carried forward between crossovers
    held = df['signal'].replace(0, np.nan)
    held = held.replace(-1, 0)
    held = held.ffill().fillna(0)

    df['position'] = held.shift(execution_lag).fillna(0)
    df['daily_return'] = df['Adj Close'].pct_change()

    df['turnover'] = df['position'].diff().abs().fillna(0)
    df['cost'] = df['turnover'] * cost_per_side

    gross = df['position'] * df['daily_return']
    df['strategy_return'] = (gross - df['cost']).fillna(0)
    df['equity_curve'] = initial_capital * (1 + df['strategy_return']).cumprod()
    return df