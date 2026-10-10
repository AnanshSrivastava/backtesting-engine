import numpy as np
import pandas as pd
import pytest

from src.metrics import sharpe_ratio, information_ratio, cagr, max_drawdown


def test_max_drawdown_known_curve():
    # peak 150 -> trough 90 is a 40% fall; the later 200 does not change that
    curve = pd.Series([100, 150, 90, 200])
    assert max_drawdown(curve) == pytest.approx(-0.4)


def test_max_drawdown_zero_for_rising_curve():
    assert max_drawdown(pd.Series([100, 110, 120, 130])) == 0


def test_cagr_two_year_growth():
    # 100 -> 121 over 730 days (~2 years) is roughly 10% a year
    idx = pd.to_datetime(["2021-01-01", "2023-01-01"])
    curve = pd.Series([100.0, 121.0], index=idx)
    assert cagr(curve) == pytest.approx(0.10, abs=1e-3)


def test_sharpe_positive_when_returns_beat_risk_free():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.002, 0.01, 500))  # ~50% a year drift
    assert sharpe_ratio(r) > 0


def test_sharpe_negative_when_returns_below_risk_free():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 500))  # zero drift < 6% risk-free
    assert sharpe_ratio(r) < 0


def test_information_ratio_only_uses_overlapping_dates():
    d1 = pd.date_range("2024-01-01", periods=6)
    d2 = pd.date_range("2024-01-04", periods=6)  # overlaps on 3 dates
    strat = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.01], index=d1)
    bench = pd.Series([0.00, 0.01, 0.02, 0.01, 0.01, 0.00], index=d2)
    overlap = d1.intersection(d2)
    active = (strat.loc[overlap] - bench.loc[overlap])
    expected = np.sqrt(252) * active.mean() / active.std()
    assert information_ratio(strat, bench) == pytest.approx(expected)


def test_information_ratio_positive_when_beating_benchmark():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=300)
    bench = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    strat = bench + rng.normal(0.001, 0.002, 300)  # consistent edge over bench
    assert information_ratio(strat, bench) > 0