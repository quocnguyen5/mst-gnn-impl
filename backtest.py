"""
Trading Simulation (Backtesting)
=================================
Implements the simple trading strategy described in the paper:

"Leveraging the stock ranking results from MST-GNN, we developed a
simple trading strategy that directly buys the top 5 and top 10
stocks with an equal budget split at the end of each trading day
and sells them at the end of the next trading day."

"We set the transaction cost as 0.03%."

Reference: Section V-F of the paper (Figs. 12 & 13).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.visualization import plot_cumulative_returns

logger = logging.getLogger(__name__)


class TradingSimulator:
    """
    Simple next-day trading strategy based on MST-GNN rankings.

    Strategy:
    1. At end of each trading day, rank stocks by predicted ranking score
    2. Buy top-K stocks with equal budget
    3. Sell all at end of next trading day
    4. Apply transaction cost
    """

    def __init__(
        self,
        top_k_stocks: List[int] = None,
        transaction_cost: float = 0.0003,
        initial_capital: float = 1_000_000.0,
    ):
        """
        Args:
            top_k_stocks: List of K values to test (default: [5, 10])
            transaction_cost: Per-trade cost as fraction (0.03% = 0.0003)
            initial_capital: Starting capital
        """
        self.top_k_stocks = top_k_stocks or [5, 10]
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

    def simulate(
        self,
        dates: List,
        stock_codes_per_day: List[List[str]],
        ranking_scores_per_day: List[np.ndarray],
        actual_returns_per_day: List[np.ndarray],
    ) -> Dict[str, Dict]:
        """
        Run trading simulation for all top-K strategies.

        Args:
            dates: Trading dates
            stock_codes_per_day: Stock codes for each day
            ranking_scores_per_day: Predicted ranking scores per day
            actual_returns_per_day: Actual stock returns per day

        Returns:
            Dict mapping strategy_name -> {
                "daily_returns": List[float],
                "cumulative_returns": List[float],
                "total_return": float,
                "annualized_return": float,
                "sharpe_ratio": float,
                "max_drawdown": float,
                "selected_stocks": List[List[str]],
            }
        """
        results = {}

        for k in self.top_k_stocks:
            strategy_name = f"Top-{k}"
            logger.info(f"Simulating {strategy_name} strategy...")

            daily_returns = []
            selected_stocks = []

            for day_idx in range(len(dates)):
                scores = ranking_scores_per_day[day_idx]
                returns = actual_returns_per_day[day_idx]
                codes = stock_codes_per_day[day_idx]

                if len(scores) < k:
                    # Not enough stocks, skip
                    daily_returns.append(0.0)
                    selected_stocks.append([])
                    continue

                # Select top-K stocks by predicted ranking score
                top_k_indices = np.argsort(-scores)[:k]
                top_k_codes = [codes[i] for i in top_k_indices]
                selected_stocks.append(top_k_codes)

                # Equal-weight portfolio return
                portfolio_return = returns[top_k_indices].mean()

                # Apply transaction costs (buy + sell)
                net_return = portfolio_return - 2 * self.transaction_cost

                daily_returns.append(float(net_return))

            # Compute cumulative returns
            cum_returns = np.cumprod(1 + np.array(daily_returns))

            # Performance metrics
            daily_returns_arr = np.array(daily_returns)
            total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
            num_days = len(daily_returns)
            annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1

            # Sharpe ratio (annualized, assuming 0 risk-free rate)
            if daily_returns_arr.std() > 0:
                sharpe = (
                    daily_returns_arr.mean()
                    / daily_returns_arr.std()
                    * np.sqrt(252)
                )
            else:
                sharpe = 0.0

            # Maximum drawdown
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (peak - cum_returns) / peak
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

            results[strategy_name] = {
                "daily_returns": daily_returns,
                "cumulative_returns": cum_returns.tolist(),
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "sharpe_ratio": float(sharpe),
                "max_drawdown": float(max_drawdown),
                "selected_stocks": selected_stocks,
            }

            logger.info(
                f"  {strategy_name}: Total Return: {total_return*100:.2f}%, "
                f"Annualized: {annualized_return*100:.2f}%, "
                f"Sharpe: {sharpe:.3f}, Max DD: {max_drawdown*100:.2f}%"
            )

        return results

    def simulate_with_baselines(
        self,
        dates: List,
        stock_codes_per_day: List[List[str]],
        actual_returns_per_day: List[np.ndarray],
        model_scores: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict]:
        """
        Run simulation comparing multiple models.

        Args:
            dates: Trading dates
            stock_codes_per_day: Stock codes per day
            actual_returns_per_day: Actual returns per day
            model_scores: Dict mapping model_name -> ranking scores per day

        Returns:
            Dict mapping "model_name_top_k" -> performance results
        """
        all_results = {}

        # Equal-weight benchmark (buy all stocks equally)
        benchmark_returns = [returns.mean() for returns in actual_returns_per_day]
        cum_benchmark = np.cumprod(1 + np.array(benchmark_returns))
        all_results["Equal-Weight Benchmark"] = {
            "daily_returns": benchmark_returns,
            "cumulative_returns": cum_benchmark.tolist(),
            "total_return": float(cum_benchmark[-1] - 1),
        }

        # Each model's strategies
        for model_name, scores_per_day in model_scores.items():
            results = self.simulate(
                dates, stock_codes_per_day, scores_per_day, actual_returns_per_day
            )
            for strategy_name, perf in results.items():
                all_results[f"{model_name} {strategy_name}"] = perf

        return all_results

    def plot_results(
        self,
        results: Dict[str, Dict],
        dates: List = None,
        save_path: str = None,
    ):
        """
        Plot cumulative return curves for all strategies.

        Reproduces Figs. 12 & 13 from the paper.
        """
        returns_dict = {}
        for name, perf in results.items():
            returns_dict[name] = perf["daily_returns"]

        plot_cumulative_returns(
            returns_dict,
            dates=dates,
            title="MST-GNN Trading Simulation: Cumulative Returns",
            save_path=save_path,
        )

    def generate_report(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Generate a summary report of all strategies."""
        rows = []
        for name, perf in results.items():
            rows.append(
                {
                    "Strategy": name,
                    "Total Return (%)": perf.get("total_return", 0) * 100,
                    "Annualized Return (%)": perf.get("annualized_return", 0) * 100,
                    "Sharpe Ratio": perf.get("sharpe_ratio", 0),
                    "Max Drawdown (%)": perf.get("max_drawdown", 0) * 100,
                }
            )
        return pd.DataFrame(rows).sort_values(
            "Total Return (%)", ascending=False
        )
