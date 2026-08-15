# notebooks/test.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fetch_data import fetch_data
from src.strategy import generate_signals

df = fetch_data("RELIANCE.NS", "2020-01-01", "2024-01-01")
if df is not None:
    print("--- verbose=True ---")
    df_signals = generate_signals(df.copy(), verbose=True)

    print("--- verbose=False ---")
    df_signals = generate_signals(df.copy(), verbose=False)
    df_signals = df_signals.dropna(subset=['SMA_long'])  # Drop rows where SMA values are NaN
    print(df_signals.tail())
else:
    print("No data for RELIANCE.NS")