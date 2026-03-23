from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from backtest import (
    compute_best_expert_in_hindsight,
    compute_equal_weight_benchmark,
    run_backtest,
)
from data_loader import compute_returns, load_prices
from experts import build_default_experts
from learner import HedgeLearner
from metrics import cumulative_wealth, summary_metrics


def print_metrics_table(title: str, metrics_dict: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics_dict.items():
        print(f"{key:>24}: {value: .6f}")


def main() -> None:
    ticker = "SPY"
    start = "2015-01-01"
    end = "2025-01-01"
    eta = 5.0
    transaction_cost = 0.0005

    df = load_prices(ticker=ticker, start=start, end=end)
    df = compute_returns(df)

    experts = build_default_experts()
    learner = HedgeLearner(eta=eta, n_experts=len(experts))

    results = run_backtest(
        df=df,
        experts=experts,
        learner=learner,
        transaction_cost=transaction_cost,
    )

    expert_returns_df = cast(pd.DataFrame, results["expert_returns_df"])
    portfolio_returns = cast(pd.Series, results["portfolio_returns"])
    portfolio_returns_net = cast(pd.Series, results["portfolio_returns_net"])
    turnover = cast(pd.Series, results["turnover"])
    weights_df = cast(pd.DataFrame, results["weights_df"])

    # Benchmarks
    buy_and_hold = expert_returns_df["buy_and_hold"]
    equal_weight = compute_equal_weight_benchmark(expert_returns_df)
    best_name, best_expert = compute_best_expert_in_hindsight(expert_returns_df)

    # Metrics
    portfolio_metrics = summary_metrics(portfolio_returns, turnover)
    portfolio_net_metrics = summary_metrics(portfolio_returns_net, turnover)
    buy_and_hold_metrics = summary_metrics(buy_and_hold)
    equal_weight_metrics = summary_metrics(equal_weight)
    best_expert_metrics = summary_metrics(best_expert)

    print_metrics_table("Hedge Portfolio (gross)", portfolio_metrics)
    print_metrics_table("Hedge Portfolio (net)", portfolio_net_metrics)
    print_metrics_table("Buy and Hold", buy_and_hold_metrics)
    print_metrics_table("Equal Weight Experts", equal_weight_metrics)
    print_metrics_table(f"Best Expert in Hindsight: {best_name}", best_expert_metrics)

    # Wealth curves
    wealth_df = pd.DataFrame(
        {
            "Hedge Gross": cumulative_wealth(portfolio_returns),
            "Hedge Net": cumulative_wealth(portfolio_returns_net),
            "Buy and Hold": cumulative_wealth(buy_and_hold),
            "Equal Weight": cumulative_wealth(equal_weight),
            f"Best Expert ({best_name})": cumulative_wealth(best_expert),
        }
    )

    plt.figure(figsize=(12, 6))
    wealth_df.plot(ax=plt.gca())
    plt.title(f"Cumulative Wealth - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Expert weights over time
    plt.figure(figsize=(12, 6))
    weights_df.plot(ax=plt.gca())
    plt.title("Expert Weights Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Turnover
    plt.figure(figsize=(12, 4))
    turnover.plot(ax=plt.gca())
    plt.title("Daily Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
