# Backtesting Engine

![tests](https://github.com/AnanshSrivastava/backtesting-engine/actions/workflows/tests.yml/badge.svg)

A Python backtesting engine for a moving-average crossover strategy on Indian equities, built from scratch: data pipeline, signal generation, position simulation, and performance analytics, compared against Nifty 50 and buy-and-hold.

The focus is on getting the mechanics right (no lookahead, adjusted prices, explicit execution timing, unit-tested core logic) and reporting results honestly, including where the strategy fails and where earlier numbers were wrong.

![Pipeline diagram](docs/pipeline_diagram.svg)

## Project Structure

```
backtesting-engine/
├── .github/workflows/tests.yml   # CI: runs pytest on every push
├── data/                         # cached OHLCV downloads (gitignored)
├── docs/                         # diagrams and charts used in this README
├── notebooks/
│   └── test.py                   # runs the full pipeline end to end
├── src/
│   ├── fetch_data.py             # yfinance download, cleaning, CSV cache
│   ├── strategy.py               # SMA crossover signal generation
│   ├── backtest.py               # signal -> position -> equity curve
│   └── metrics.py                # Sharpe, Information Ratio, CAGR, max drawdown
├── tests/                        # pytest suite
├── pytest.ini
├── requirements.txt
└── README.md
```

## Pipeline

1. **`fetch_data.py`** downloads OHLCV data with `yfinance`, flattens the MultiIndex columns, and checks for empty results. Each (ticker, start, end) download is cached to `data/` and reused on later runs; pass `use_cache=False` to force a fresh download. All calculations use `Adj Close` so splits and dividends do not create fake price jumps.
2. **`strategy.py`** computes a 20-day and a 50-day SMA and flags the exact day the short SMA crosses the long one (`+1` golden cross, `-1` death cross), using a `.diff()` on the above/below state rather than the state itself.
3. **`backtest.py`** turns the sparse crossover signals into a continuous holding/flat position, delays it so trades execute at the following session's close, applies the daily returns, and compounds them into an equity curve.
4. **`metrics.py`** computes annualised Sharpe ratio (against a risk-free rate), Information Ratio (against Nifty 50), CAGR, and maximum drawdown.

## Execution Timing

A crossover is only known at the close of day T. The backtest therefore enters the position at the **close of the following session (T+1)** and earns returns from there. This is one day more conservative than the common vectorised shortcut of shifting by one bar, which implicitly assumes a fill at the same close that produced the signal.

The choice matters a lot for this strategy (see Results), so it is pinned by unit tests: the tests fail if the delay is changed to 0 or 1.

## Testing and CI

```bash
python -m pytest -q
```

The suite covers the execution-delay logic on a hand-built frame, known-answer checks for drawdown and CAGR, Sharpe and Information Ratio behaviour (including date alignment), and the data cache (using a fake downloader, so no network is needed). GitHub Actions runs the same tests on every push.

## Results: RELIANCE.NS, SMA 20/50, 2020 to 2024

No transaction costs or slippage are modelled yet, so every figure below is gross and preliminary.

| Metric | Strategy (fills next close, current) | Strategy (fills signal-day close, earlier version) | Buy and hold RELIANCE.NS | Buy and hold Nifty 50 |
|---|---|---|---|---|
| Total return | -1.30% | 13.45% | 169.80% | 126.60% |
| CAGR | -0.34% | 3.38% | not computed | not computed |
| Max drawdown | -49.56% | -41.78% | -21.68% | not computed |
| Sharpe (risk-free 6%) | -0.2281 | -0.0430 | not computed | not computed |
| Information Ratio vs Nifty 50 | -0.8859 | -0.7407 | not computed | not computed |

All columns start on 2020-03-12, the first day after the 50-day SMA warm-up, so the comparison is like for like. The earlier-version column was computed with the same start for the strategy and the stock.

What the verified numbers say:

- Over this window the strategy roughly broke even, while buy-and-hold RELIANCE.NS returned 169.80% and Nifty 50 returned 126.60%.
- **The result is highly sensitive to fill timing.** Delaying execution by one extra day moved total return from 13.45% to -1.30%. For a lagging crossover strategy that fragility is itself a finding, and it means single-run numbers should not be trusted without a confidence interval (planned below).
- A likely driver of the large drawdown is whipsaw: after a strong trend ends, repeated small crossovers produce a string of small losing trades. This explanation is a hypothesis. The peak and trough dates are being re-traced under the current execution timing.

![Equity Curve](docs/equity_curve.png)

### Evaluation notes

- The strategy and the buy-and-hold figures start on the first day after the 50-day SMA warm-up (2020-03-12), so buy-and-hold max drawdown excludes the February to March 2020 crash.
- An earlier version of this README quoted a Nifty 50 return of 78.38%. That was measured from 2020-01-02, before the strategy's start date, so it understated the benchmark. The 126.60% above uses the same start as the backtest.
- Results for other windows (2022 to 2024 and two 2018 windows) came from earlier versions of the code and have been withdrawn until re-run.
- The risk-free rate is a fixed 6% approximation, not historical government bond yields.

### Corrections log

- An earlier version labelled the benchmark-relative ratio as "Sharpe". It is the Information Ratio. Both are now reported separately.
- An earlier version filled trades at the signal-day close. Execution is now at the following session's close, which changed the 2020 to 2024 total return from 13.45% to -1.30%.
- An earlier version compared the strategy against a Nifty return measured from a different start date. All columns now share one start date.
- An early run had a leading NaN that propagated through the compounded equity curve. It is fixed and covered by a test.

## Setup

```bash
git clone https://github.com/AnanshSrivastava/backtesting-engine.git
cd backtesting-engine

python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# .\venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

## Usage

```bash
python notebooks/test.py
```

This fetches the stock and benchmark, runs the crossover strategy, simulates the backtest, and prints CAGR, Sharpe, Information Ratio, and max drawdown against buy-and-hold.

## Limitations and Roadmap

- No transaction costs or slippage yet. Adding them is the next step, since gross results flatter every strategy.
- Single asset, long-only, one fixed parameter set (20/50).
- Planned: block-bootstrap confidence interval on Sharpe, plus ADF and autocorrelation checks on returns.
- Planned: a walk-forward logistic-regression strategy compared against SMA and buy-and-hold after costs, with every variant tested counted and reported.

## Tech

Python, pandas, NumPy, yfinance, matplotlib, pytest, GitHub Actions