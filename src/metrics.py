import numpy as np
import pandas as pd
#Sharpe ratio is a measure of risk-adjusted return, which helps investors understand how much excess return they are receiving 
# for the extra volatility endured by holding a riskier asset. It is calculated as the difference between the returns of the investment
# and the risk-free return, divided by the standard deviation of the investment returns.
def sharpe_ratio(strategy_returns, risk_free_rate=0.06, N=252):
    """
    Calculate the annualised Sharpe ratio: risk-adjusted return relative
    to a risk-free rate.

    Parameters:
        strategy_returns (pd.Series): Daily returns of the strategy.
        risk_free_rate (float): Annual risk-free rate (default is 6%).
        N (int): Trading periods per year (default 252).

    Returns:
        float: The Sharpe ratio of the strategy.
    """
    daily_risk_free_rate = risk_free_rate / N
    excess_returns = strategy_returns - daily_risk_free_rate
    excess_returns = excess_returns.dropna()  # Drop NaN values to avoid issues in calculations

    return np.sqrt(N) * excess_returns.mean() / excess_returns.std()


def information_ratio(strategy_returns, benchmark_returns, N=252):
    """
    Calculate the annualised Information Ratio: consistency of a strategy's
    outperformance relative to a benchmark.

    Parameters:
        strategy_returns (pd.Series): Daily returns of the strategy.
        benchmark_returns (pd.Series): Daily returns of the benchmark.
        N (int): Trading periods per year (default 252).

    Returns:
        float: The Information Ratio of the strategy vs. the benchmark.
    """
    # align both series on matching dates
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
    aligned.columns = ['strategy', 'benchmark']
    active_returns = aligned['strategy'] - aligned['benchmark']
    active_returns = active_returns.dropna()  # Drop NaN values to avoid issues in calculations

    return np.sqrt(N) * active_returns.mean() / active_returns.std()

#cagr is a measure of the mean annual growth rate of an investment over a specified period of time longer than one year.
def cagr(equity_curve):
    """
    Calculate the Compound Annual Growth Rate from an equity curve.
    
    Parameters:
        equity_curve (Series): equity values indexed by date
    
    Returns:
        float: annualised CAGR
    """
    starting_value = equity_curve.iloc[0]
    ending_value = equity_curve.iloc[-1]
    
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    
    return (ending_value / starting_value) ** (1 / years) - 1

#max_drawdown is a measure of the largest single drop from peak to trough in the value of a portfolio, before a new peak is achieved.
def max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown (largest peak-to-trough decline)
    from an equity curve.
    
    Returns:
        float: maximum drawdown as a negative percentage (e.g. -0.25 = 25% decline)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()