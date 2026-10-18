import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import run_backtest
from src.evaluation import evaluation_window
from src.ml_strategy import (
    build_features,
    build_target,
    position_state,
    rsi,
    signals_from_probabilities,
    walk_forward_probabilities,
)


def make_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame({"Adj Close": price}, index=idx)


def test_target_horizon_alignment():
    df = pd.DataFrame({"Adj Close": [100, 101, 100, 102, 101, 103.0]})
    t1 = build_target(df, 1).tolist()
    t2 = build_target(df, 2).tolist()
    assert t1[:5] == [1, 0, 1, 0, 1] and np.isnan(t1[5])
    assert t2[:4] == [0, 1, 0, 1] and np.isnan(t2[4]) and np.isnan(t2[5])


def test_features_do_not_use_future_prices():
    df = make_df()
    altered = df.copy()
    altered.iloc[400:, 0] *= 1.7
    a, b = build_features(df), build_features(altered)
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])


def test_rsi_bounds_and_trend():
    up = pd.Series(np.arange(1, 80, dtype=float))
    r = rsi(up).dropna()
    assert (r >= 0).all() and (r <= 100).all() and r.iloc[-1] > 99
    noisy = make_df()["Adj Close"]
    r2 = rsi(noisy).dropna()
    assert (r2 >= 0).all() and (r2 <= 100).all()


@pytest.mark.parametrize("horizon", [1, 2])
def test_walk_forward_predictions_ignore_the_future(horizon):
    df = make_df()
    cfg = dict(train_window=120, retrain_every=10, min_train=60)
    cut = 450
    altered = df.copy()
    rng = np.random.default_rng(1)
    altered.iloc[cut + 1:, 0] *= rng.uniform(0.5, 1.5, len(df) - cut - 1)

    out = []
    for d in (df, altered):
        p = walk_forward_probabilities(
            build_features(d), build_target(d, horizon),
            eval_start=df.index[300], horizon=horizon, **cfg,
        )
        out.append(p)
    pd.testing.assert_series_equal(out[0].iloc[: cut + 1], out[1].iloc[: cut + 1])
    assert out[0].iloc[300: cut + 1].notna().any()


def test_no_predictions_before_eval_start():
    df = make_df()
    p = walk_forward_probabilities(
        build_features(df), build_target(df, 1),
        eval_start=df.index[300], horizon=1,
        train_window=120, retrain_every=10, min_train=60,
    )
    assert p.iloc[:300].isna().all()


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_state_becomes_position_with_lag(lag):
    df = make_df(200)
    rng = np.random.default_rng(3)
    probs = pd.Series(rng.uniform(0, 1, len(df)), index=df.index)
    probs.iloc[:20] = np.nan
    state = position_state(probs)
    bt = df.copy()
    bt["signal"] = signals_from_probabilities(probs)
    out = run_backtest(bt, execution_lag=lag)
    pd.testing.assert_series_equal(
        out["position"], state.shift(lag).fillna(0).astype(float), check_names=False
    )


def test_initial_long_state_is_not_lost():
    df = make_df(50)
    probs = pd.Series(0.9, index=df.index)   # long from the first row
    bt = df.copy()
    bt["signal"] = signals_from_probabilities(probs)
    out = run_backtest(bt, execution_lag=1)
    assert out["position"].iloc[1:].eq(1.0).all()


def test_prediction_at_t_earns_return_at_t_plus_lag():
    lag = 2
    df = make_df(100)
    t0 = 50
    probs = pd.Series(0.0, index=df.index)
    probs.iloc[t0] = 1.0                      # long signal on a single day
    bt = df.copy()
    bt["signal"] = signals_from_probabilities(probs)
    out = run_backtest(bt, execution_lag=lag, cost_per_side=0.0)
    ret = df["Adj Close"].pct_change()
    assert out["position"].iloc[t0 + lag] == 1.0
    assert out["strategy_return"].iloc[t0 + lag] == pytest.approx(ret.iloc[t0 + lag])
    assert out["position"].drop(out.index[t0 + lag]).eq(0).all()
    # and the label at t0 refers to exactly that return
    assert build_target(df, lag).iloc[t0] == float(ret.iloc[t0 + lag] > 0)


def test_evaluation_window_uses_first_day_as_base():
    df = make_df(200)
    bt = df.copy()
    bt["signal"] = 0
    out = run_backtest(bt, execution_lag=1)
    win = evaluation_window(out, df.index[50], 100000)
    assert win["equity_curve"].iloc[0] == 100000
    assert win["strategy_return"].iloc[0] == 0 and win.index[0] == df.index[50]