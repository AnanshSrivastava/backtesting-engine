"""Statistical diagnostics and run logging for the backtester."""
import json
import math
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

# Anchored to the repo root (src/..), so it does not depend on the cwd.
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "run_registry.jsonl"


def _clean(returns):
    return pd.Series(returns).dropna().to_numpy(dtype=float)


def annualized_sharpe(returns, risk_free_rate=0.06, periods_per_year=252):
    """Annualized Sharpe on daily returns (same convention as metrics.sharpe_ratio)."""
    values = _clean(returns)
    if len(values) < 2:
        return float("nan")
    excess = values - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def adf_check(returns):
    """
    ADF unit-root test on daily strategy returns.

    Sanity check only: daily returns are almost always stationary, so a
    rejection says nothing about whether the strategy has an edge.
    """
    values = _clean(returns)
    if len(values) < 20:
        raise ValueError("Need at least 20 non-null returns for the ADF check")
    if np.ptp(values) == 0:
        raise ValueError("Returns are constant (strategy never traded?)")

    with warnings.catch_warnings():
        # statsmodels warns about a future change in adfuller's return type
        warnings.simplefilter("ignore", FutureWarning)
        statistic, p_value, used_lags, observations, critical, _ = adfuller(
            values, autolag="AIC"
        )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lags": int(used_lags),
        "observations": int(observations),
        "critical_values": {k: float(v) for k, v in critical.items()},
        "reject_unit_root_at_5pct": bool(p_value < 0.05),
    }


def autocorrelation_check(returns, max_lag=20):
    """
    ACF with per-lag 95% white-noise bounds plus a joint Ljung-Box test.

    Per-lag flags are uncorrelated-noise bounds: at 5% each, ~max_lag*0.05
    lags will be flagged by chance. The Ljung-Box p-value is the joint test.
    """
    values = _clean(returns)
    if len(values) <= max_lag:
        raise ValueError("Need more observations than max_lag")

    correlations = acf(values, nlags=max_lag, fft=True)
    bound = 1.96 / np.sqrt(len(values))
    lags = [
        {
            "lag": lag,
            "autocorrelation": float(correlations[lag]),
            "outside_approx_95pct_bound": bool(abs(correlations[lag]) > bound),
        }
        for lag in range(1, max_lag + 1)
    ]
    lb = acorr_ljungbox(values, lags=[max_lag], return_df=True).iloc[0]
    return {
        "observations": int(len(values)),
        "approx_95pct_bound": float(bound),
        "expected_false_flags": float(0.05 * max_lag),
        "ljung_box": {
            "lag": max_lag,
            "statistic": float(lb["lb_stat"]),
            "p_value": float(lb["lb_pvalue"]),
            "reject_white_noise_at_5pct": bool(lb["lb_pvalue"] < 0.05),
        },
        "lags": lags,
    }


def bootstrap_sharpe_ci(
    returns,
    risk_free_rate=0.06,
    periods_per_year=252,
    block_length=10,
    n_bootstrap=5000,
    confidence=0.95,
    seed=42,
):
    """
    Circular moving-block bootstrap percentile CI for annualized Sharpe (vectorized).

    Blocks keep short-range dependence. Resamples with zero variance give NaN
    and are dropped (counted in n_nan).
    """
    values = _clean(returns)
    n = len(values)
    if n < 2:
        raise ValueError("Need at least two non-null returns")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and sample size")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_bootstrap, blocks_needed))
    idx = (starts[:, :, None] + np.arange(block_length)) % n
    idx = idx.reshape(n_bootstrap, -1)[:, :n]

    excess = values[idx] - risk_free_rate / periods_per_year
    sd = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        estimates = np.sqrt(periods_per_year) * excess.mean(axis=1) / sd
    estimates[sd == 0] = np.nan

    valid = estimates[np.isfinite(estimates)]
    if len(valid) == 0:
        raise ValueError("All bootstrap resamples had zero variance")

    alpha = 1 - confidence
    lower, upper = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": annualized_sharpe(values, risk_free_rate, periods_per_year),
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "n_nan": int(n_bootstrap - len(valid)),
        "seed": seed,
    }


def _json_safe(obj):
    """Recursively convert numpy types and non-finite floats (-> None)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def append_run_record(record, path=DEFAULT_REGISTRY):
    """Append one JSON line per backtest run; NaN/inf are stored as null."""
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    full_record = _json_safe(
        {
            "run_id": str(uuid.uuid4()),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }
    )
    with registry_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(full_record, allow_nan=False) + "\n")
    return full_record