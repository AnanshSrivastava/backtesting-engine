# notebooks/test.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fetch_data import fetch_data

df = fetch_data("RELIANCE.NS", "2023-01-01", "2024-01-01")
if df is not None:
    print(df.head())
else:
    print("No data for RELIANCE.NS")