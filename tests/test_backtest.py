import pandas as pd
import pytest

from src.backtest import run_backtest


def make_df():
    # buy signal confirmed at the close of row 1, sell signal at the close of row 5
    return pd.DataFrame({
        "signal":    [0, 1, 0, 0, 0, -1, 0, 0],
        "Adj Close": [100, 105, 110, 121, 121, 133.1, 133.1, 133.1],
    })


def test_position_executes_at_following_close():
    out = run_backtest(make_df())
    # Signal is confirmed at the close of row 1. We trade at the NEXT session's
    # close (row 2), so the first return we earn is row 3's (close 2 -> close 3).
    # Sell confirmed row 5 -> we exit at row 6's close, last return earned is row 5's.
    assert out["position"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_nothing_is_earned_on_or_before_the_signal_day():
    out = run_backtest(make_df())
    assert (out["strategy_return"].iloc[:3] == 0).all()


def test_flat_days_earn_nothing():
    out = run_backtest(make_df())
    # row 7 is flat again after the exit
    assert out["strategy_return"].iloc[7] == 0


def test_equity_curve_compounds_only_while_holding():
    out = run_backtest(make_df())
    # earned: row 3 (+10%), row 4 (0%), row 5 (+10%) -> 100000 * 1.1 * 1.1
    assert out["equity_curve"].iloc[-1] == pytest.approx(121000)


def test_equity_curve_has_no_nans():
    out = run_backtest(make_df())
    assert out["equity_curve"].isnull().sum() == 0