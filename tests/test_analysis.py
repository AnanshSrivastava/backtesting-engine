import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    adf_check,
    annualized_sharpe,
    append_run_record,
    autocorrelation_check,
    bootstrap_sharpe_ci,
)
from src.backtest import run_backtest
from src.strategy import generate_signals


def test_sharpe_matches_manual_formula():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    ex = r - 0.06 / 252
    assert annualized_sharpe(r) == pytest.approx(np.sqrt(252) * ex.mean() / ex.std(ddof=1))


def test_sharpe_constant_series_is_nan():
    assert np.isnan(annualized_sharpe([0.0] * 50, risk_free_rate=0.0))


def test_bootstrap_is_deterministic_and_brackets_estimate():
    r = np.random.default_rng(1).normal(0.0008, 0.01, 800)
    a = bootstrap_sharpe_ci(r, n_bootstrap=1000, seed=7)
    b = bootstrap_sharpe_ci(r, n_bootstrap=1000, seed=7)
    assert a == b
    assert a["lower"] < a["estimate"] < a["upper"]


def test_bootstrap_ci_straddles_zero_for_pure_noise():
    r = np.random.default_rng(2).normal(0.0, 0.01, 1000)
    ci = bootstrap_sharpe_ci(r, risk_free_rate=0.0, n_bootstrap=2000)
    assert ci["lower"] < 0 < ci["upper"]


def test_bootstrap_handles_mostly_flat_returns():
    r = np.zeros(300)
    r[100:110] = 0.01
    ci = bootstrap_sharpe_ci(r, n_bootstrap=500)
    assert np.isfinite(ci["lower"]) and np.isfinite(ci["upper"])


def test_adf_random_walk_vs_white_noise():
    rng = np.random.default_rng(3)
    noise = rng.normal(size=1000)
    assert adf_check(noise)["reject_unit_root_at_5pct"]
    assert not adf_check(np.cumsum(noise))["reject_unit_root_at_5pct"]


def test_acf_detects_ar1_and_ljung_box_agrees():
    rng = np.random.default_rng(4)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.4 * x[t - 1] + rng.normal()
    out = autocorrelation_check(x, max_lag=10)
    assert out["lags"][0]["outside_approx_95pct_bound"]
    assert out["ljung_box"]["reject_white_noise_at_5pct"]


def test_run_record_survives_nan_and_numpy_types(tmp_path):
    p = tmp_path / "reg.jsonl"
    append_run_record({"sharpe": float("nan"), "n": np.int64(3), "ok": np.bool_(True)}, path=p)
    append_run_record({"sharpe": 1.0}, path=p)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["sharpe"] is None and rows[0]["n"] == 3 and rows[0]["ok"] is True
    assert rows[0]["run_id"] != rows[1]["run_id"]


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_no_lookahead_future_prices_do_not_change_past_positions(lag):
    rng = np.random.default_rng(5)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    base = pd.DataFrame({"Adj Close": price})
    altered = base.copy()
    altered.loc[250:, "Adj Close"] *= rng.uniform(0.5, 1.5, len(altered) - 250)

    pos = []
    for d in (base, altered):
        d = generate_signals(d.copy(), verbose=False)
        pos.append(run_backtest(d, execution_lag=lag)["position"])
    # position[t] may only depend on prices up to t - lag
    pd.testing.assert_series_equal(pos[0].iloc[: 250 + lag], pos[1].iloc[: 250 + lag])