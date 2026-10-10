# notebooks/plot_equity_curve.py
"""
Generates the equity curve chart: strategy vs. buy-and-hold vs. Nifty 50.
Run this after your existing test.py pipeline (fetch -> signals -> backtest).
Saves the chart to docs/equity_curve.png for the README.
"""
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fetch_data import fetch_data
from src.strategy import generate_signals
from src.backtest import run_backtest

TICKER = "RELIANCE.NS"
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INITIAL_CAPITAL = 100000

df = fetch_data(TICKER, START_DATE, END_DATE)
benchmark_df = fetch_data(BENCHMARK_TICKER, START_DATE, END_DATE)

df_signals = generate_signals(df.copy(), verbose=False)
df_signals = df_signals.dropna(subset=['SMA_long'])
backtest_results = run_backtest(df_signals)

# Normalise buy-and-hold series to the same starting capital, for a fair visual comparison
stock_bh = INITIAL_CAPITAL * (backtest_results['Adj Close'] / backtest_results['Adj Close'].iloc[0])
# Start Nifty on the same day as the backtest, so all three lines are comparable
bench = benchmark_df.loc[backtest_results.index[0]:]
nifty_bh = INITIAL_CAPITAL * (bench['Adj Close'] / bench['Adj Close'].iloc[0])

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.plot(backtest_results.index, backtest_results['equity_curve'], label='SMA Crossover Strategy', color='#f97316', linewidth=1.8)
ax.plot(backtest_results.index, stock_bh, label=f'Buy & Hold {TICKER}', color='#6366f1', linewidth=1.3, alpha=0.85)
ax.plot(nifty_bh.index, nifty_bh, label='Buy & Hold Nifty 50', color='#10b981', linewidth=1.3, alpha=0.85)

ax.axhline(INITIAL_CAPITAL, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

start_label = backtest_results.index[0].strftime('%Y-%m-%d')
ax.set_title(f'{TICKER}: Strategy vs. Buy & Hold (from {start_label}, after 50-day SMA warm-up)',fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (₹)')
ax.legend(loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "docs"), exist_ok=True)
save_path = os.path.join(os.path.dirname(__file__), "..", "docs", "equity_curve.png")
plt.savefig(save_path, dpi=150)
print(f"Saved chart to {save_path}")

plt.show()