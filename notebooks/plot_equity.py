# notebooks/plot_equity_curve.py
"""
Equity curve + drawdown chart: strategy vs buy-and-hold vs Nifty 50, over the
same evaluation window and with the same settings as notebooks/test.py.

Also prints a ready-to-paste results table and the peak/trough dates of each
max drawdown. Keep the constants below in sync with test.py.
Saves the chart to docs/equity_curve.png for the README.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import run_backtest
from src.fetch_data import fetch_data
from src.metrics import cagr, information_ratio, max_drawdown, sharpe_ratio
from src.strategy import generate_signals

TICKER = "RELIANCE.NS"
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
WARMUP_DAYS = 120
SMA_SHORT = 20
SMA_LONG = 50
COST_PER_SIDE = 0.001
INITIAL_CAPITAL = 100000
RISK_FREE_RATE = 0.06
HEADLINE_LAG = 2     # trade at the next close
OPTIMISTIC_LAG = 1   # trade at the signal-day close

COLORS = {
    "strategy": "#f97316",
    "strategy_opt": "#fdba74",
    "stock": "#6366f1",
    "nifty": "#10b981",
}


def evaluation_window(df_signals, lag):
    """Backtest on the full history, then keep START_DATE onward (same as test.py)."""
    results = run_backtest(
        df_signals.copy(),
        initial_capital=INITIAL_CAPITAL,
        execution_lag=lag,
        cost_per_side=COST_PER_SIDE,
    )
    window = results.loc[START_DATE:].copy()
    window.iloc[0, window.columns.get_loc("strategy_return")] = 0.0
    window.iloc[0, window.columns.get_loc("turnover")] = 0.0
    window["equity_curve"] = INITIAL_CAPITAL * (1 + window["strategy_return"]).cumprod()
    return window


def drawdown_series(equity):
    return equity / equity.cummax() - 1


def peak_trough(equity):
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return peak.date(), trough.date(), dd.min()


# ---------------------------------------------------------------- data
fetch_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
df = fetch_data(TICKER, fetch_start, END_DATE)
benchmark_df = fetch_data(BENCHMARK_TICKER, fetch_start, END_DATE)
if df is None or benchmark_df is None:
    raise SystemExit("Missing price data; cannot plot.")

df_signals = generate_signals(df.copy(), SMA_short=SMA_SHORT, SMA_long=SMA_LONG, verbose=False)
df_signals = df_signals.dropna(subset=["SMA_long"])
if df_signals.index[0] > pd.Timestamp(START_DATE):
    raise SystemExit("Warm-up too short: increase WARMUP_DAYS.")

headline = evaluation_window(df_signals, HEADLINE_LAG)
optimistic = evaluation_window(df_signals, OPTIMISTIC_LAG)

first, last = headline.index[0], headline.index[-1]
stock_prices = headline["Adj Close"]
nifty_prices = benchmark_df.loc[first:last, "Adj Close"]

curves = {
    f"SMA crossover (lag {HEADLINE_LAG}, after costs)": headline["equity_curve"],
    f"SMA crossover (lag {OPTIMISTIC_LAG}, optimistic)": optimistic["equity_curve"],
    f"Buy & hold {TICKER}": INITIAL_CAPITAL * stock_prices / stock_prices.iloc[0],
    "Buy & hold Nifty 50": INITIAL_CAPITAL * nifty_prices / nifty_prices.iloc[0],
}
names = list(curves)
styles = [
    dict(color=COLORS["strategy"], linewidth=1.9),
    dict(color=COLORS["strategy_opt"], linewidth=1.3, linestyle="--"),
    dict(color=COLORS["stock"], linewidth=1.3, alpha=0.9),
    dict(color=COLORS["nifty"], linewidth=1.3, alpha=0.9),
]

# ---------------------------------------------------------------- chart
fig, (ax_eq, ax_dd) = plt.subplots(
    2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
)

for name, style in zip(names, styles):
    ax_eq.plot(curves[name].index, curves[name], label=name, **style)
ax_eq.axhline(INITIAL_CAPITAL, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax_eq.set_ylabel("Portfolio value (₹)")
ax_eq.set_title(
    f"{TICKER}: SMA {SMA_SHORT}/{SMA_LONG} vs buy & hold "
    f"({first.date()} to {last.date()}, {COST_PER_SIDE * 1e4:.0f} bps per side)",
    fontsize=12, fontweight="bold",
)
ax_eq.legend(loc="best", frameon=False)

for name, style in zip(names, styles):
    if "optimistic" in name:
        continue  # keep the drawdown panel readable
    dd = drawdown_series(curves[name])
    ax_dd.plot(dd.index, dd, label=name, **style)
    if name == names[0]:
        ax_dd.fill_between(dd.index, dd, 0, color=COLORS["strategy"], alpha=0.15)
ax_dd.set_ylabel("Drawdown from running peak")
ax_dd.set_xlabel("Date")
ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

for ax in (ax_eq, ax_dd):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(docs_dir, exist_ok=True)
save_path = os.path.join(docs_dir, "equity_curve.png")
plt.savefig(save_path, dpi=150)
print(f"Saved chart to {save_path}\n")

# ---------------------------------------------------------------- numbers
nifty_returns_full = benchmark_df["Adj Close"].pct_change()


def strategy_row(label, window):
    eq, r = window["equity_curve"], window["strategy_return"].iloc[1:]
    return (
        label,
        eq.iloc[-1] / eq.iloc[0] - 1,
        cagr(eq),
        max_drawdown(eq),
        sharpe_ratio(r, risk_free_rate=RISK_FREE_RATE),
        information_ratio(r, nifty_returns_full),
    )


def buy_hold_row(label, prices):
    r = prices.pct_change().iloc[1:]
    return (
        label,
        prices.iloc[-1] / prices.iloc[0] - 1,
        cagr(prices),
        max_drawdown(prices),
        sharpe_ratio(r, risk_free_rate=RISK_FREE_RATE),
        information_ratio(r, nifty_returns_full) if label != "Buy & hold Nifty 50" else float("nan"),
    )


rows = [
    strategy_row(f"SMA crossover, lag {HEADLINE_LAG}", headline),
    strategy_row(f"SMA crossover, lag {OPTIMISTIC_LAG}", optimistic),
    buy_hold_row(f"Buy & hold {TICKER}", stock_prices),
    buy_hold_row("Buy & hold Nifty 50", nifty_prices),
]


def fmt(v, kind):
    if pd.isna(v):
        return "n/a"
    return f"{v:.2%}" if kind == "pct" else f"{v:.4f}"


print("| Strategy | Total return | CAGR | Max drawdown | Sharpe | Info ratio vs Nifty |")
print("|---|---|---|---|---|---|")
for label, tot, c, mdd, sh, ir in rows:
    print(f"| {label} | {fmt(tot, 'pct')} | {fmt(c, 'pct')} | {fmt(mdd, 'pct')} | "
          f"{fmt(sh, 'num')} | {fmt(ir, 'num')} |")

print("\nMax drawdown peak -> trough:")
for name in (names[0], names[1], names[2], names[3]):
    p, t, d = peak_trough(curves[name])
    print(f"  {name}: {p} -> {t} ({d:.2%})")

plt.show()