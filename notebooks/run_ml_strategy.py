# notebooks/run_ml_strategy.py
"""
Walk-forward logistic-regression strategy vs the SMA crossover and buy-and-hold,
all measured over the same window, after costs, with bootstrap Sharpe CIs.

SMA and baseline numbers use exactly the same data fetch as notebooks/test.py,
so they reproduce its output. The ML model uses a longer history so it has
training data before the evaluation window starts.
"""
import hashlib
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (
    adf_check,
    append_run_record,
    autocorrelation_check,
    bootstrap_sharpe_ci,
)
from src.backtest import run_backtest
from src.evaluation import evaluation_window
from src.fetch_data import fetch_data
from src.metrics import cagr, information_ratio, max_drawdown, sharpe_ratio
from src.ml_strategy import (
    DEFAULT_CONFIG,
    build_features,
    build_target,
    signals_from_probabilities,
    walk_forward_probabilities,
)
from src.strategy import generate_signals

TICKER = "RELIANCE.NS"
BENCHMARK_TICKER = "^NSEI"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
WARMUP_DAYS = 120               # same as test.py (SMA comparator)
ML_HISTORY_START = "2016-01-01" # extra history so the model can train before START_DATE
SMA_SHORT, SMA_LONG = 20, 50
COST_PER_SIDE = 0.001
INITIAL_CAPITAL = 100000
RISK_FREE_RATE = 0.06
LAGS = (2, 1)                   # 2 = headline (next close), 1 = optimistic
STRATEGY_NAME = "logreg walk-forward v1"
CFG = DEFAULT_CONFIG


def summarize(equity, returns, nifty_returns, n_trades=None, exposure=None):
    sharpe = sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE)
    ci = bootstrap_sharpe_ci(returns, risk_free_rate=RISK_FREE_RATE,
                             block_length=10, n_bootstrap=5000, seed=42)
    assert abs(ci["estimate"] - sharpe) < 1e-9, "Sharpe mismatch"
    return {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": cagr(equity),
        "max_dd": max_drawdown(equity),
        "sharpe": sharpe,
        "ci": ci,
        "info_ratio": None if nifty_returns is None else information_ratio(returns, nifty_returns),
        "n_trades": n_trades,
        "exposure": exposure,
    }


def summarize_window(window, nifty_returns):
    return summarize(
        window["equity_curve"], window["strategy_return"].iloc[1:], nifty_returns,
        n_trades=int((window["turnover"] > 0).sum()),
        exposure=float(window["position"].mean()),
    )


def safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:   # e.g. constant returns when a strategy never trades
        print(f"  (skipped {fn.__name__}: {exc})")
        return None


def main():
    # ------------------------------------------------------------ data
    fetch_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    df_sma_raw = fetch_data(TICKER, fetch_start, END_DATE)
    bench_raw = fetch_data(BENCHMARK_TICKER, fetch_start, END_DATE)
    df_ml = fetch_data(TICKER, ML_HISTORY_START, END_DATE)
    if df_sma_raw is None or bench_raw is None or df_ml is None:
        print("Missing price data.")
        return

    nifty_returns = bench_raw["Adj Close"].pct_change()
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df_ml).values.tobytes()
    ).hexdigest()[:12]

    # ------------------------------------------------ SMA comparator (as in test.py)
    sma = generate_signals(df_sma_raw.copy(), SMA_short=SMA_SHORT, SMA_long=SMA_LONG, verbose=False)
    sma = sma.dropna(subset=["SMA_long"])

    # ------------------------------------------------ baselines over the window
    prices = sma.loc[START_DATE:, "Adj Close"]
    bench_prices = bench_raw.loc[prices.index[0]:prices.index[-1], "Adj Close"]
    rows = {}
    rows["Buy & hold " + TICKER] = summarize(
        INITIAL_CAPITAL * prices / prices.iloc[0], prices.pct_change().iloc[1:], nifty_returns)
    rows["Buy & hold Nifty 50"] = summarize(
        INITIAL_CAPITAL * bench_prices / bench_prices.iloc[0],
        bench_prices.pct_change().iloc[1:], None)   # no IR of Nifty vs itself

    features = build_features(df_ml, CFG["return_lags"])

    for lag in LAGS:
        # ---- SMA
        sma_win = evaluation_window(
            run_backtest(sma.copy(), initial_capital=INITIAL_CAPITAL,
                         execution_lag=lag, cost_per_side=COST_PER_SIDE),
            START_DATE, INITIAL_CAPITAL)
        rows[f"SMA 20/50, lag {lag}"] = summarize_window(sma_win, nifty_returns)

        # ---- ML: walk-forward probabilities, target horizon = execution lag
        target = build_target(df_ml, lag)
        probs = walk_forward_probabilities(features, target, START_DATE, lag, **CFG)
        bt_in = df_ml[["Adj Close"]].copy()
        bt_in["signal"] = signals_from_probabilities(probs, CFG["threshold"])

        net = evaluation_window(
            run_backtest(bt_in.copy(), initial_capital=INITIAL_CAPITAL,
                         execution_lag=lag, cost_per_side=COST_PER_SIDE),
            START_DATE, INITIAL_CAPITAL)
        gross = evaluation_window(
            run_backtest(bt_in.copy(), initial_capital=INITIAL_CAPITAL,
                         execution_lag=lag, cost_per_side=0.0),
            START_DATE, INITIAL_CAPITAL)
        assert net.index.equals(sma_win.index), "ML and SMA windows differ"

        stats = summarize_window(net, nifty_returns)
        sharpe_gross = sharpe_ratio(gross["strategy_return"].iloc[1:], risk_free_rate=RISK_FREE_RATE)
        rows[f"{STRATEGY_NAME}, lag {lag}"] = stats

        # ---- classification quality on the evaluation window
        m = probs.notna() & target.notna() & (probs.index >= pd.Timestamp(START_DATE))
        y, p = target[m].astype(int), probs[m]
        accuracy = float(((p > CFG["threshold"]).astype(int) == y).mean())
        base_rate = float(y.mean())
        majority = max(base_rate, 1 - base_rate)
        band = 1.96 * (0.25 / len(y)) ** 0.5
        auc = float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")

        returns = net["strategy_return"].iloc[1:]
        adf_result = safe(adf_check, returns)
        acf_result = safe(autocorrelation_check, returns, max_lag=20)

        record = append_run_record({
            "ticker": TICKER, "benchmark_ticker": BENCHMARK_TICKER,
            "start_date": START_DATE, "end_date": END_DATE,
            "window_first_day": str(net.index[0].date()),
            "window_last_day": str(net.index[-1].date()),
            "ml_history_start": ML_HISTORY_START,
            "strategy": STRATEGY_NAME, "model_config": CFG,
            "features": list(features.columns),
            "execution_lag": lag, "cost_per_side": COST_PER_SIDE,
            "initial_capital": INITIAL_CAPITAL, "risk_free_rate": RISK_FREE_RATE,
            "data_hash": data_hash,
            "n_trades": stats["n_trades"], "exposure": stats["exposure"],
            "strategy_cagr": stats["cagr"], "strategy_total_return": stats["total_return"],
            "strategy_max_drawdown": stats["max_dd"],
            "sharpe_ratio": stats["sharpe"], "sharpe_no_costs": sharpe_gross,
            "information_ratio": stats["info_ratio"],
            "accuracy": accuracy, "majority_class_rate": majority, "auc": auc,
            "n_predictions": int(len(y)),
            "adf": adf_result, "autocorrelation": acf_result,
            "sharpe_bootstrap_ci": stats["ci"],
        })

        print(f"--- {STRATEGY_NAME} | lag {lag} | {net.index[0].date()} to {net.index[-1].date()} ---")
        print(f"Directional accuracy:     {accuracy:.2%}  (always-majority-class {majority:.2%}, "
              f"chance band +/-{band:.2%}, n={len(y)})")
        print(f"AUC:                      {auc:.4f}  (0.5 = no skill)")
        print(f"Sharpe without costs:     {sharpe_gross:.4f}   with costs: {stats['sharpe']:.4f}")
        print(f"Position changes:         {stats['n_trades']}  (exposure {stats['exposure']:.1%})")
        print(f"Run ID:                   {record['run_id']}\n")

    # ------------------------------------------------ comparison table
    print("=== Comparison (after costs, same window) ===")
    print("| Strategy | Total return | CAGR | Max DD | Sharpe | Sharpe 95% CI | Info ratio | Changes | Invested |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, s in rows.items():
        ir = "n/a" if s["info_ratio"] is None else f"{s['info_ratio']:.4f}"
        ch = "n/a" if s["n_trades"] is None else str(s["n_trades"])
        ex = "n/a" if s["exposure"] is None else f"{s['exposure']:.1%}"
        print(f"| {name} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_dd']:.2%} | "
              f"{s['sharpe']:.4f} | [{s['ci']['lower']:.2f}, {s['ci']['upper']:.2f}] | {ir} | {ch} | {ex} |")


if __name__ == "__main__":
    main()