from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from backtest import (
    compute_best_expert_in_hindsight,
    compute_equal_weight_benchmark,
    run_backtest,
)
from data_loader import compute_returns, load_prices
from experts import build_default_experts
from learner import HedgeLearner
from metrics import cumulative_wealth, summary_metrics


def format_metrics_table(title: str, metrics_dict: dict[str, float]) -> str:
    lines = [f"\n{title}", "-" * len(title)]
    for key, value in metrics_dict.items():
        lines.append(f"{key:>24}: {value: .6f}")
    return "\n".join(lines)


def build_metrics_report(sections: list[tuple[str, dict[str, float]]]) -> str:
    return "\n".join(
        format_metrics_table(title, metrics_dict) for title, metrics_dict in sections
    ).lstrip()


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

    metrics_sections = [
        ("Hedge Portfolio (gross)", portfolio_metrics),
        ("Hedge Portfolio (net)", portfolio_net_metrics),
        ("Buy and Hold", buy_and_hold_metrics),
        ("Equal Weight Experts", equal_weight_metrics),
        (f"Best Expert in Hindsight: {best_name}", best_expert_metrics),
    ]
    metrics_report = build_metrics_report(metrics_sections)
    print(metrics_report)

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

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    wealth_df.plot(ax=axes[0])
    axes[0].set_title(f"Cumulative Wealth - {ticker}")
    axes[0].set_ylabel("Wealth")
    axes[0].grid(True)

    weights_df.plot(ax=axes[1])
    axes[1].set_title("Expert Weights Over Time")
    axes[1].set_ylabel("Weight")
    axes[1].grid(True)

    turnover.plot(ax=axes[2], color="black")
    axes[2].set_title("Daily Turnover")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Turnover")
    axes[2].grid(True)

    fig.tight_layout()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = f"{ticker.lower()}_{start}_{end}"
    pdf_path = output_dir / f"{report_prefix}_plots_{timestamp}.pdf"
    txt_path = output_dir / f"{report_prefix}_metrics_{timestamp}.txt"

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)

    txt_path.write_text(metrics_report + "\n", encoding="utf-8")
    plt.close(fig)

    print(f"\nPlot salvati in: {pdf_path}")
    print(f"Output testuale salvato in: {txt_path}")


if __name__ == "__main__":
    main()
