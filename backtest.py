from __future__ import annotations

import numpy as np
import pandas as pd

from experts import Expert
from learner import HedgeLearner


def run_backtest(
    df: pd.DataFrame,
    experts: list[Expert],
    learner: HedgeLearner,
    transaction_cost: float = 0.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Run the online learning backtest.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['price', 'return'] indexed by date.
    experts : list[Expert]
        Trading experts.
    learner : HedgeLearner
        Online learner.
    transaction_cost : float
        Cost multiplier applied to daily turnover.

    Returns
    -------
    dict containing:
        - signals_df
        - expert_returns_df
        - weights_df
        - portfolio_returns
        - portfolio_returns_net
        - turnover
    """
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")

    n_experts = len(experts)
    if n_experts == 0:
        raise ValueError("At least one expert is required")

    max_lookback = max(expert.min_history for expert in experts)
    if len(df) <= max_lookback:
        raise ValueError("Not enough rows in dataframe for the selected experts")

    dates = []
    signals_records = []
    expert_returns_records = []
    weights_records = []
    portfolio_returns = []
    portfolio_returns_net = []
    turnover_values = []

    prev_weights = learner.get_weights()

    # We start at t = max_lookback so that every expert has enough history
    for t in range(max_lookback, len(df)):
        date = df.index[t]
        history = df.iloc[:t]  # only up to t-1
        current_return = df["return"].iloc[t]

        signals = np.array([expert.signal(history) for expert in experts], dtype=float)
        expert_returns = signals * current_return

        current_weights = learner.get_weights()
        gross_portfolio_return = float(np.dot(current_weights, expert_returns))

        turnover = float(np.sum(np.abs(current_weights - prev_weights)))
        net_portfolio_return = gross_portfolio_return - transaction_cost * turnover

        # Save records before updating
        dates.append(date)
        signals_records.append(signals.copy())
        expert_returns_records.append(expert_returns.copy())
        weights_records.append(current_weights.copy())
        portfolio_returns.append(gross_portfolio_return)
        portfolio_returns_net.append(net_portfolio_return)
        turnover_values.append(turnover)

        # Update weights for next day
        learner.update(expert_returns)
        prev_weights = current_weights

    expert_names = [expert.name for expert in experts]

    signals_df = pd.DataFrame(signals_records, index=dates, columns=expert_names)
    expert_returns_df = pd.DataFrame(
        expert_returns_records, index=dates, columns=expert_names
    )
    weights_df = pd.DataFrame(weights_records, index=dates, columns=expert_names)

    portfolio_returns_series = pd.Series(
        portfolio_returns, index=dates, name="portfolio_return"
    )
    portfolio_returns_net_series = pd.Series(
        portfolio_returns_net, index=dates, name="portfolio_return_net"
    )
    turnover_series = pd.Series(turnover_values, index=dates, name="turnover")

    return {
        "signals_df": signals_df,
        "expert_returns_df": expert_returns_df,
        "weights_df": weights_df,
        "portfolio_returns": portfolio_returns_series,
        "portfolio_returns_net": portfolio_returns_net_series,
        "turnover": turnover_series,
    }


def compute_equal_weight_benchmark(expert_returns_df: pd.DataFrame) -> pd.Series:
    """
    Equal-weight portfolio over experts.
    """
    return expert_returns_df.mean(axis=1)


def compute_best_expert_in_hindsight(
    expert_returns_df: pd.DataFrame,
) -> tuple[str, pd.Series]:
    """
    Return the name and return series of the best expert in hindsight.
    """
    cumulative_returns = (1.0 + expert_returns_df).prod(axis=0) - 1.0

    # The explicit str() conversion suggests idxmax() might return a
    # non-string type, but this could indicate an underlying type inconsistency.
    # ----
    # Consider ensuring expert_returns_df column names are strings
    # from the outset rather than converting at the point of use.
    best_name = str(cumulative_returns.idxmax())
    return best_name, expert_returns_df[best_name].copy()
