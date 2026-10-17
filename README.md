# Backtesting Engine

![tests](https://github.com/AnanshSrivastava/backtesting-engine/actions/workflows/tests.yml/badge.svg)

A Python backtesting engine for a moving-average crossover strategy on Indian equities, built from scratch: data pipeline, signal generation, position simulation with transaction costs, performance analytics, and statistical checks on the results, compared against Nifty 50 and buy-and-hold.

The focus is on getting the mechanics right (no lookahead, adjusted prices, explicit execution timing, unit-tested core logic) and reporting results honestly, including where the strategy fails and where earlier numbers were wrong.

![Pipeline diagram](docs/pipeline_diagram.svg)

## Project Structure

```
backtesting-engine/
├── .github/workflows/tests.yml   # CI: runs pytest on every push
├── data/                         # cached OHLCV downloads and run registry (gitignored)
├── docs/                         # diagrams and charts used in this README
├── notebooks/
│   ├── test.py                   # full pipeline: both execution lags, stats, run registry
│   └── plot_equity.py            # equity curve + drawdown chart -> docs/equity_curve.png
├── src/
│   ├── fetch_data.py             # yfinance download, cleaning, CSV cache
│   ├── strategy.py               # SMA crossover signal generation
│   ├── backtest.py               # signal -> position -> costs -> equity curve
│   ├── metrics.py                # Sharpe, Information Ratio, CAGR, max drawdown
│   └── analysis.py               # ADF, autocorrelation, bootstrap Sharpe CI, run registry
├── tests/                        # pytest suite
├── pytest.ini
├── requirements.txt
└── README.md
```

## Pipeline

1. **`fetch_data.py`** downloads OHLCV data with `yfinance`, flattens the MultiIndex columns, and checks for empty results. Each (ticker, start, end) download is cached to `data/` and reused on later runs; pass `use_cache=False` to force a fresh download. All calculations use `Adj Close` so splits and dividends do not create fake price jumps.
2. **`strategy.py`** computes a 20-day and a 50-day SMA and flags the exact day the short SMA crosses the long one (`+1` golden cross, `-1` death cross), using a `.diff()` on the above/below state rather than the state itself. No signal is produced until two consecutive valid long-SMA values exist, so the warm-up period cannot fake a crossover.
3. **`backtest.py`** turns the sparse crossover signals into a continuous holding/flat position, delays it by `execution_lag` sessions, applies the daily returns, deducts transaction costs on every position change, and compounds the result into an equity curve.
4. **`metrics.py`** computes annualised Sharpe ratio (against a risk-free rate), Information Ratio (against Nifty 50), CAGR, and maximum drawdown.
5. **`analysis.py`** runs statistical checks on the strategy returns (see below) and appends every run to a registry, so the number of variants tested is on record.

## Execution Timing and Costs

A crossover is only known at the close of day T. Two execution assumptions are supported, and both are reported:

- **Lag 2 (headline):** enter at the close of the following session (T+1) and earn returns from there. This is the conservative choice.
- **Lag 1 (optimistic bound):** enter at the signal-day close itself. This is the common vectorised shortcut of shifting by one bar, and it implicitly assumes a fill at the same close that produced the signal.

Lags below 1 would trade on information not yet available and are rejected. The default delay is covered by unit tests, including a property test that changes future prices and checks that earlier positions do not move.

Costs are 10 bps of traded value per side, charged on every change of position (a round trip pays twice). Slippage and market impact are not modelled.

## Evaluation Window

The 50-day SMA needs about two months of history before it is valid. The pipeline fetches 120 calendar days before the start date, runs signals and the backtest on the full history, and then measures everything (strategy, buy-and-hold, Nifty 50) over exactly the same window, starting on the stated start date. The first day of the window is the base, so no return from the warm-up period leaks into the results.

## Testing and CI

```bash
python -m pytest -q
```

The suite covers the execution-delay logic on a hand-built frame, known-answer checks for drawdown and CAGR, Sharpe and Information Ratio behaviour (including date alignment), the data cache (using a fake downloader, so no network is needed), and the statistical checks: Sharpe against its formula, a deterministic and well-behaved bootstrap, ADF and autocorrelation on synthetic series with known answers, and the run registry. GitHub Actions runs the same tests on every push.

## Results: RELIANCE.NS, SMA 20/50, 2020-01-01 to 2023-12-29

Single stock, single window, one parameter set. Costs of 10 bps per side are included. Risk-free rate 6%.

| Metric | Strategy, fills next close (lag 2, headline) | Strategy, fills signal-day close (lag 1, optimistic) | Buy and hold RELIANCE.NS | Buy and hold Nifty 50 |
|---|---|---|---|---|
| Total return | -3.94% | 10.43% | 89.98% | 78.38% |
| CAGR | -1.00% | 2.52% | 17.44% | 15.60% |
| Max drawdown | -50.66% | -43.04% | -44.08% | -38.44% |
| Sharpe (risk-free 6%) | -0.2737 | -0.0945 | 0.4863 | 0.5350 |
| Sharpe 95% CI (block bootstrap) | [-1.33, 0.75] | [-1.19, 0.94] | not computed | not computed |
| Information Ratio vs Nifty 50 | -0.7090 | -0.5672 | 0.1958 | n/a |
| Position changes / time invested | 27 / 51.4% | 27 / 51.5% | n/a | n/a |

![Equity curve and drawdown](docs/equity_curve.png)

*Top: portfolio value. Bottom: drawdown from the running peak.*

What the numbers say:

- After costs the strategy lost money over this window, while buy-and-hold RELIANCE.NS returned 89.98% and Nifty 50 returned 78.38%. The strategy does not demonstrate an edge.
- Its maximum drawdown (-50.66%) is deeper than buy-and-hold's (-44.08%) even though it was invested only about half the time. The two come from different places. Buy-and-hold's drawdown is the COVID crash (2020-01-17 to 2020-03-23), which the strategy sat out. The strategy's runs from a peak on 2020-09-16 to a trough on 2022-12-23: a slide of more than two years rather than a single crash. That pattern fits whipsaw, where a lagging crossover keeps buying after rises and selling after falls in a choppy market, but it is still a hypothesis. Confirming it needs a trade-by-trade profit and loss breakdown, which has not been done.
- Risk-adjusted, the strategy's Sharpe (-0.27) is far below buy-and-hold RELIANCE.NS (0.49) and Nifty 50 (0.54). Nifty's Sharpe is slightly above the stock's despite a lower return; its crash drawdown was also shallower (-38.44% against -44.08%).
- **The result is sensitive to fill timing.** Moving execution by one day changes total return from -3.94% to +10.43%, a gap far larger than the cost of 27 position changes (about 2.7% in total). For a lagging crossover strategy that fragility is itself a finding.

### Statistical checks

- **Sharpe confidence intervals.** A circular moving-block bootstrap (block length 10, 5,000 resamples, fixed seed) gives 95% intervals that include zero at both lags and overlap almost completely. The data cannot separate this strategy from zero skill, and it cannot show the strategy is good either. With roughly 4 years and 27 position changes, wide intervals are expected. Blocks of 10 days are shorter than the typical holding period, so the intervals are probably still a little too narrow.
- **Autocorrelation.** A Ljung-Box test over 20 lags gives p = 0.021 (lag 2) and p = 0.008 (lag 1), with individual lags 5, 8 and 16 outside the white-noise bounds at lag 2 (about one lag is expected by chance at 20 lags). The two lags are the same strategy shifted by a day, so this is one weak finding, not two. Being invested or flat in clusters produces autocorrelation mechanically, so it is not read as exploitable structure.
- **ADF test.** Rejects a unit root at both lags (p < 0.0001). Daily returns are almost always stationary, so this is a sanity check and says nothing about edge.

### Evaluation notes

- Strategy, buy-and-hold and Nifty 50 are all measured from the same first day, 2020-01-01, so the February to March 2020 crash is included in every column. The last day, 2023-12-29, is the final trading day before the end date.
- The risk-free rate is a fixed 6% approximation, not historical government bond yields. Flat days earn 0 in the backtest but have the risk-free rate subtracted in the Sharpe calculation, which slightly penalises the strategy.
- Results for other windows (2022 to 2024 and two 2018 windows) came from earlier versions of the code and have been withdrawn until re-run.
- Every run is logged to `data/run_registry.jsonl` with its parameters, data fingerprint and statistics. Variants tested so far: 1 (SMA 20/50) at 2 execution lags.

### Corrections log

- An earlier version labelled the benchmark-relative ratio as "Sharpe". It is the Information Ratio. Both are now reported separately.
- An earlier version filled trades at the signal-day close. Execution is now at the following session's close by default, with the old behaviour available as the optimistic lag-1 bound.
- An earlier version compared the strategy against a Nifty return measured from a different start date. All columns now share one start date.
- An early run had a leading NaN that propagated through the compounded equity curve. It is fixed and covered by a test.
- **An earlier version evaluated everything from 2020-03-12, the first day after the 50-day SMA warm-up.** That start sits just after the COVID crash, so it inflated buy-and-hold (169.80% and 126.60% for the stock and Nifty, against 89.98% and 78.38% over the full window) and left the crash out of buy-and-hold's drawdown (-21.68%, against -44.08%). The pipeline now fetches warm-up data before the start date so all figures cover the full stated window. An intermediate correction had changed the Nifty figure from 78.38% to 126.60%; that fixed a start-date mismatch but not this window problem.
- An earlier version modelled no transaction costs, so the earlier gross figures (13.45% and -1.30% total return) are superseded.

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
python notebooks/test.py              # runs both execution lags and prints the results and statistical checks
python notebooks/plot_equity.py # saves docs/equity_curve.png and prints a results table
```

`test.py` fetches the stock and benchmark (with warm-up), runs the crossover strategy at lag 2 and lag 1, and prints CAGR, Sharpe with a bootstrap confidence interval, Information Ratio, max drawdown, and the autocorrelation and ADF checks against buy-and-hold and Nifty 50. Each run is appended to the run registry.

## Limitations and Roadmap

- Slippage and market impact are not modelled, and the 10 bps cost is a flat approximation rather than the actual Indian brokerage, STT and stamp duty schedule.
- Single asset, long-only, one fixed parameter set (20/50), one four-year window. Statistical power is low, so no conclusion about the strategy in general should be drawn.
- Planned: a walk-forward logistic-regression strategy compared against SMA and buy-and-hold after costs, with every variant tested counted and reported.

## Tech

Python, pandas, NumPy, statsmodels, yfinance, matplotlib, pytest, GitHub Actions