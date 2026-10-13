import pandas as pd


def generate_signals(df, SMA_short=20, SMA_long=50, verbose=True):
    """
    Generate SMA crossover signals.

    Adds SMA_short, SMA_long, short_over_long and signal columns.
    signal is +1 on the day the short SMA crosses above the long SMA,
    -1 on the day it crosses below, and 0 otherwise.

    No signal is produced until two consecutive days of valid long-SMA
    values exist: with fewer, there is no previous state to cross from.
    Without this, a rising warm-up period fakes a golden cross on the
    first day the long SMA becomes available.

    Parameters:
        df (DataFrame): must contain an 'Adj Close' column.
        SMA_short (int): short window length.
        SMA_long (int): long window length.
        verbose (bool): print the crossover table and counts.

    Returns:
        DataFrame with the new columns added.
    """
    df['SMA_short'] = df['Adj Close'].rolling(window=SMA_short).mean()
    df['SMA_long'] = df['Adj Close'].rolling(window=SMA_long).mean()
    df['short_over_long'] = df['SMA_short'] > df['SMA_long']

    signal = df['short_over_long'].astype(int).diff()

    # A crossover needs a valid state on both today and yesterday.
    valid = df['SMA_long'].notna()
    signal[~(valid & valid.shift(1, fill_value=False))] = 0
    df['signal'] = signal

    if verbose:
        crossovers = df[df['signal'].isin([1, -1])]
        print(crossovers[['Adj Close', 'SMA_short', 'SMA_long', 'signal']])
        for _, row in crossovers.iterrows():
            print("Buy signal generated." if row['signal'] == 1 else "Sell signal generated.")
        print(df['signal'].tail(5))

    return df