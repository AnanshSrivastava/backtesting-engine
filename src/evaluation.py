"""Shared evaluation-window logic."""


def evaluation_window(results, start_date, initial_capital):
    """
    Keep results from start_date onward and rebuild the equity curve.

    The backtest runs on the full history (warm-up included) so the position on
    the first evaluation day is already correct. The first day of the window is
    the base: returns earned before it belong to the warm-up, so its return and
    turnover are zeroed and equity is compounded from there.
    """
    window = results.loc[start_date:].copy()
    window.iloc[0, window.columns.get_loc("strategy_return")] = 0.0
    window.iloc[0, window.columns.get_loc("turnover")] = 0.0
    window["equity_curve"] = initial_capital * (1 + window["strategy_return"]).cumprod()
    return window