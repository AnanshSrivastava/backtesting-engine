import numpy as np
import pandas as pd

from src.backtest import run_backtest


def test_buy_signal_does_not_earn_returns_before_assumed_fill():
    # The buy signal is known at index 1's close. The assumed fill is index 2's
    # close, so the strategy first earns the return ending at index 3.
    prices = [100, 200, 400, 800, 1600]
    signals = [0, 1, 0, 0, 0]
    frame = pd.DataFrame(
        {"Adj Close": prices, "signal": signals},
        index=pd.date_range("2025-01-01", periods=len(prices)),
    )

    result = run_backtest(frame)

    np.testing.assert_array_equal(result["position"].to_numpy(), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(
        result["strategy_return"].to_numpy(), [0, 0, 0, 1, 1]
    )


def test_sell_signal_keeps_position_until_assumed_fill():
    # The sell signal is known at index 3's close. We remain exposed through
    # the return ending at index 4, then are flat for returns from index 5.
    prices = [100, 100, 100, 100, 200, 400, 800]
    signals = [1, 0, 0, -1, 0, 0, 0]
    frame = pd.DataFrame(
        {"Adj Close": prices, "signal": signals},
        index=pd.date_range("2025-01-01", periods=len(prices)),
    )

    result = run_backtest(frame)

    np.testing.assert_array_equal(
        result["position"].to_numpy(), [0, 0, 1, 1, 1, 0, 0]
    )
    assert result["strategy_return"].iloc[4] == 1.0
    assert result["strategy_return"].iloc[5] == 0.0
