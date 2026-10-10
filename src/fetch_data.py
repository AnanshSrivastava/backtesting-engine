"""Fetch OHLCV data from Yahoo Finance, with a local CSV cache."""
import os
import pandas as pd
import yfinance as yf

# Absolute path to <repo>/data, independent of where the script is run from
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def fetch_data(ticker, start_date, end_date, use_cache=True, data_dir=DATA_DIR):
    """
    Return a clean OHLCV DataFrame for one ticker, or None on failure.

    If a cached CSV for this exact (ticker, start, end) exists and use_cache
    is True, it is loaded instead of hitting the network. Pass use_cache=False
    to force a fresh download (which also overwrites the cache file).
    """
    cache_path = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")

    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        print(f"Error occurred while fetching data for {ticker}: {e}")
        return None

    if df.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return None

    df.columns = df.columns.get_level_values(0)

    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(cache_path)
    return df


def fetch_multiple(tickers, start_date, end_date, use_cache=True):
    """Fetch several tickers. Returns {ticker: DataFrame}, skipping failures."""
    data = {}
    for ticker in tickers:
        df = fetch_data(ticker, start_date, end_date, use_cache=use_cache)
        if df is not None:
            data[ticker] = df
    return data