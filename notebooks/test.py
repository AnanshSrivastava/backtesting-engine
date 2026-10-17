# notebooks/test.py
"""
End-to-end pipeline: fetch data (with a warm-up period), generate signals,
run the backtest at several execution lags, report metrics measured over
exactly START_DATE..END_DATE, and log every run to the registry.
"""
import hashlib
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (
    adf_check,
    append_run_record,
    autocorrelation_check,
    bootstrap_sharpe_ci,
)
from src.backtest import run_backtest
from src.fetch_data import fetch_data
from src.metrics import cagr, information_ratio, max_drawdown, sharpe_ratio
from src.strategy import generate_signals

TICKER = "RELIANCE.NS"
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2020-01-01"   # evaluation window start
END_DATE = "2024-01-01"
WARMUP_DAYS = 120           # calendar days fetched before START_DATE so the
                            # 50-day SMA is already valid on START_DATE
SMA_SHORT = 20
SMA_LONG = 50
COST_PER_SIDE = 0.001
INITIAL_CAPITAL = 100000
RISK_FREE_RATE = 0.06
# Lag 2 = trade at the next close (conservative, headline).
# Lag 1 = trade at the signal-day close (optimistic bound).
EXECUTION_LAGS = (2, 1)


def run_one(df_signals, benchmark_df, lag, baselines, data_hash):
    """Backtest at one execution lag over the evaluation window, log the run."""
    # Run on the full history (warm-up included) so the position on the first
    # evaluation day is already correct.
    results = run_backtest(
        df_signals.copy(),  # run_backtest mutates its input
        initial_capital=INITIAL_CAPITAL,
        execution_lag=lag,
        cost_per_side=COST_PER_SIDE,
    )

    # Keep only the evaluation window. Its first day is the base: returns
    # earned before it belong to the warm-up, so zero them and rebuild equity.
    window = results.loc[START_DATE:].copy()
    window.iloc[0, window.columns.get_loc("strategy_return")] = 0.0
    window.iloc[0, window.columns.get_loc("turnover")] = 0.0
    window["equity_curve"] = INITIAL_CAPITAL * (1 + window["strategy_return"]).cumprod()

    equity = window["equity_curve"]
    returns = window["strategy_return"].iloc[1:]

    strategy_cagr = cagr(equity)
    strategy_max_dd = max_drawdown(equity)
    strategy_total_return = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE)
    info_ratio = information_ratio(returns, benchmark_df["daily_return"])

    adf_result = adf_check(returns)
    acf_result = autocorrelation_check(returns, max_lag=20)
    sharpe_ci = bootstrap_sharpe_ci(
        returns,
        risk_free_rate=RISK_FREE_RATE,
        block_length=10,
        n_bootstrap=5000,
        seed=42,
    )
    # The two Sharpe implementations must agree.
    assert abs(sharpe_ci["estimate"] - sharpe) < 1e-9, "Sharpe mismatch"

    n_trades = int((window["turnover"] > 0).sum())
    exposure = float(window["position"].mean())

    record = append_run_record({
        "ticker": TICKER,
        "benchmark_ticker": BENCHMARK_TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "window_first_day": str(window.index[0].date()),
        "window_last_day": str(window.index[-1].date()),
        "warmup_days": WARMUP_DAYS,
        "strategy": "SMA crossover",
        "sma_short": SMA_SHORT,
        "sma_long": SMA_LONG,
        "execution_lag": lag,
        "cost_per_side": COST_PER_SIDE,
        "initial_capital": INITIAL_CAPITAL,
        "risk_free_rate": RISK_FREE_RATE,
        "data_hash": data_hash,
        "n_trades": n_trades,
        "exposure": exposure,
        "strategy_cagr": float(strategy_cagr),
        "strategy_total_return": float(strategy_total_return),
        "strategy_max_drawdown": float(strategy_max_dd),
        "buy_hold_max_drawdown": float(baselines["buy_hold_max_dd"]),
        "buy_hold_total_return": float(baselines["buy_hold_return"]),
        "nifty_total_return": float(baselines["nifty_return"]),
        "sharpe_ratio": float(sharpe),
        "information_ratio": float(info_ratio),
        "adf": adf_result,
        "autocorrelation": acf_result,
        "sharpe_bootstrap_ci": sharpe_ci,
    })

    significant_lags = [
        item["lag"] for item in acf_result["lags"]
        if item["outside_approx_95pct_bound"]
    ]
    lb = acf_result["ljung_box"]

    print(f"--- {TICKER} | {window.index[0].date()} to {window.index[-1].date()} "
          f"| execution_lag={lag} ---")
    print(f"Strategy CAGR:            {strategy_cagr:.2%}")
    print(f"Sharpe Ratio:             {sharpe:.4f}")
    print(f"Strategy Max Drawdown:    {strategy_max_dd:.2%}")
    print(f"Buy & Hold Max Drawdown:  {baselines['buy_hold_max_dd']:.2%}")
    print(f"Strategy Total Return:    {strategy_total_return:.2%}")
    print(f"Buy & Hold Total Return:  {baselines['buy_hold_return']:.2%}")
    print(f"Nifty 50 Total Return:    {baselines['nifty_return']:.2%}")
    print(f"Information Ratio:        {info_ratio:.4f}")
    print(f"Trades / exposure:        {n_trades} / {exposure:.1%}")
    print(f"ADF p-value:              {adf_result['p_value']:.4f}")
    print(f"Significant ACF lags:     {significant_lags if significant_lags else 'none'}"
          f" (~{acf_result['expected_false_flags']:.0f} expected by chance)")
    print(f"Ljung-Box p-value:        {lb['p_value']:.4f}")
    print(f"Bootstrap Sharpe 95% CI:  [{sharpe_ci['lower']:.4f}, {sharpe_ci['upper']:.4f}]")
    print(f"Run ID:                   {record['run_id']}\n")

    return {
        "lag": lag,
        "cagr": strategy_cagr,
        "sharpe": sharpe,
        "ci_low": sharpe_ci["lower"],
        "ci_high": sharpe_ci["upper"],
        "max_dd": strategy_max_dd,
        "total_return": strategy_total_return,
    }


def main():
    fetch_start = (
        pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)
    ).strftime("%Y-%m-%d")

    df = fetch_data(TICKER, fetch_start, END_DATE)
    benchmark_df = fetch_data(BENCHMARK_TICKER, fetch_start, END_DATE)

    if df is None:
        print(f"No data for {TICKER}")
        return
    if benchmark_df is None:
        print(f"No data for benchmark {BENCHMARK_TICKER}")
        return

    # Fingerprint of the raw downloaded data, computed once.
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:12]

    benchmark_df["daily_return"] = benchmark_df["Adj Close"].pct_change()

    df_signals = generate_signals(
        df.copy(), SMA_short=SMA_SHORT, SMA_long=SMA_LONG, verbose=False
    )
    df_signals = df_signals.dropna(subset=["SMA_long"])
    if df_signals.index[0] > pd.Timestamp(START_DATE):
        raise ValueError(
            "Warm-up too short: SMA_long not valid by START_DATE; "
            "increase WARMUP_DAYS"
        )

    # Lag-independent baselines over the exact evaluation window.
    prices = df_signals.loc[START_DATE:, "Adj Close"]
    bench = benchmark_df.loc[prices.index[0]:prices.index[-1], "Adj Close"]
    baselines = {
        "buy_hold_return": prices.iloc[-1] / prices.iloc[0] - 1,
        "buy_hold_max_dd": max_drawdown(prices),
        "nifty_return": bench.iloc[-1] / bench.iloc[0] - 1,
    }

    summary = [
        run_one(df_signals, benchmark_df, lag, baselines, data_hash)
        for lag in EXECUTION_LAGS
    ]

    print("=== Lag comparison ===")
    print(f"{'lag':>3} {'CAGR':>9} {'Sharpe':>8} {'95% CI':>18} {'MaxDD':>9} {'TotRet':>9}")
    for s in summary:
        ci = f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
        print(f"{s['lag']:>3} {s['cagr']:>9.2%} {s['sharpe']:>8.3f} {ci:>18} "
              f"{s['max_dd']:>9.2%} {s['total_return']:>9.2%}")


if __name__ == "__main__":
    main()