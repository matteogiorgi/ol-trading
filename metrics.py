from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def cumulative_wealth(returns: pd.Series, initial_wealth: float = 1.0) -> pd.Series:
    wealth = initial_wealth * (1.0 + returns).cumprod()
    return wealth


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Simple annualized Sharpe ratio.
    risk_free_rate is annualized.
    """
    if returns.empty:
        return np.nan

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std(ddof=1)

    if std == 0 or np.isnan(std):
        return np.nan

    return excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(returns: pd.Series, initial_wealth: float = 1.0) -> float:
    wealth = cumulative_wealth(returns, initial_wealth=initial_wealth)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown.min()


def average_turnover(turnover: pd.Series) -> float:
    if turnover.empty:
        return np.nan
    return turnover.mean()


def summary_metrics(
    returns: pd.Series, turnover: pd.Series | None = None
) -> dict[str, float]:
    metrics = {
        "cumulative_return": cumulative_wealth(returns).iloc[-1] - 1.0,
        "sharpe_ratio": sharpe_ratio(returns),
        "annualized_volatility": annualized_volatility(returns),
        "max_drawdown": max_drawdown(returns),
    }

    if turnover is not None:
        metrics["average_turnover"] = average_turnover(turnover)

    return metrics
