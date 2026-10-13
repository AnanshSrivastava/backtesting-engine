import pandas as pd
import pytest

from src.backtest import run_backtest


def make_df():
    # buy signal confirmed at the close of row 1, sell signal at the close of row 5
    return pd.DataFrame({
        "signal":    [0, 1, 0, 0, 0, -1, 0, 0],
        "Adj Close": [100, 105, 110, 121, 121, 133.1, 133.1, 133.1],
    })


def test_default_lag_executes_at_following_close():
    out = run_backtest(make_df())
    # Confirmed at the close of row 1, trade at the close of row 2, so the first
    # return earned is row 3's. Sell confirmed row 5, exit at row 6's close.
    assert out["position"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_lag_one_executes_at_signal_close():
    out = run_backtest(make_df(), execution_lag=1)
    assert out["position"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_lag_below_one_is_rejected():
    with pytest.raises(ValueError):
        run_backtest(make_df(), execution_lag=0)


def test_nothing_is_earned_on_or_before_the_signal_day():
    out = run_backtest(make_df())
    assert (out["strategy_return"].iloc[:3] == 0).all()


def test_flat_days_earn_nothing():
    out = run_backtest(make_df())
    assert out["strategy_return"].iloc[7] == 0


def test_equity_curve_compounds_only_while_holding():
    out = run_backtest(make_df())
    # earned: row 3 (+10%), row 4 (0%), row 5 (+10%) -> 100000 * 1.1 * 1.1
    assert out["equity_curve"].iloc[-1] == pytest.approx(121000)


def test_lag_one_earns_different_equity():
    # shorter lag captures the 110/105 move too; guards against lag being ignored
    out = run_backtest(make_df(), execution_lag=1)
    assert out["equity_curve"].iloc[-1] == pytest.approx(100000 * (110 / 105) * 1.1 * 1.1)


def test_cost_is_charged_on_entry_and_exit():
    out = run_backtest(make_df(), cost_per_side=0.01)
    # entry day (row 3): +10% - 1%; exit day (row 7): 0% - 1%
    assert out["strategy_return"].iloc[3] == pytest.approx(0.09)
    assert out["strategy_return"].iloc[7] == pytest.approx(-0.01)
    assert out["equity_curve"].iloc[-1] == pytest.approx(100000 * 1.09 * 1.1 * 0.99)


def test_zero_cost_matches_gross_result():
    out = run_backtest(make_df(), cost_per_side=0.0)
    assert out["equity_curve"].iloc[-1] == pytest.approx(121000)


def test_costs_never_help():
    free = run_backtest(make_df(), cost_per_side=0.0)["equity_curve"].iloc[-1]
    costly = run_backtest(make_df(), cost_per_side=0.002)["equity_curve"].iloc[-1]
    assert costly < free


def test_no_trades_means_no_costs():
    df = pd.DataFrame({"signal": [0] * 5, "Adj Close": [100, 101, 102, 103, 104]})
    out = run_backtest(df, cost_per_side=0.05)
    assert out["cost"].sum() == 0
    assert out["equity_curve"].iloc[-1] == pytest.approx(100000)


def test_equity_curve_has_no_nans():
    out = run_backtest(make_df(), cost_per_side=0.001)
    assert out["equity_curve"].isnull().sum() == 0