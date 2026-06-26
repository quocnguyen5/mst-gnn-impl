"""
Evaluation Metrics
==================
Implements the metrics used in the paper:
- Accuracy (classification)
- Precision (classification)
- DAMRR — Daily Average Mean Reciprocal Rank (Eq. 20)

Reference: Section V-B of the paper.
"""

import numpy as np
import torch
from typing import List, Dict


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Standard classification accuracy.

    Args:
        predictions: Predicted class labels (num_samples,)
        labels: Ground truth labels (num_samples,)

    Returns:
        Accuracy as float in [0, 1]
    """
    return (predictions == labels).mean()


def precision(predictions: np.ndarray, labels: np.ndarray, pos_label: int = 1) -> float:
    """
    Precision for the positive class (price goes up).

    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        pos_label: Positive class label (1 = up)

    Returns:
        Precision as float in [0, 1]
    """
    predicted_positive = (predictions == pos_label)
    if predicted_positive.sum() == 0:
        return 0.0
    true_positive = (predictions == pos_label) & (labels == pos_label)
    return true_positive.sum() / predicted_positive.sum()


def mean_reciprocal_rank(
    ranking_scores: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """
    Mean Reciprocal Rank (MRR) for a single day.

    For each stock ranked by predicted score, compute 1/rank of its
    actual position in the true return ordering.

    Args:
        ranking_scores: Predicted ranking scores (num_stocks,)
        actual_returns: Actual returns (num_stocks,)

    Returns:
        MRR value
    """
    n = len(ranking_scores)
    if n == 0:
        return 0.0

    # Sort by predicted scores (descending)
    predicted_order = np.argsort(-ranking_scores)

    # Sort by actual returns (descending) → true ranking
    true_order = np.argsort(-actual_returns)
    true_rank = np.zeros(n, dtype=int)
    for rank, idx in enumerate(true_order):
        true_rank[idx] = rank + 1  # 1-indexed rank

    # MRR: average of 1/true_rank for each position in predicted order
    reciprocal_ranks = 1.0 / true_rank[predicted_order]
    return reciprocal_ranks.mean()


def daily_average_mrr(
    daily_scores: List[np.ndarray],
    daily_returns: List[np.ndarray],
) -> float:
    """
    Daily Average Mean Reciprocal Rank (DAMRR).

    Eq. (20) in the paper:
        DAMRR = (1/|D|) Σ_{d∈D} MRR_d

    "We define the daily average of mean reciprocal rank (DAMRR)
    to evaluate the ranking performance."

    Args:
        daily_scores: List of predicted scores per day
        daily_returns: List of actual returns per day

    Returns:
        DAMRR value
    """
    if not daily_scores:
        return 0.0

    mrr_values = []
    for scores, returns in zip(daily_scores, daily_returns):
        mrr = mean_reciprocal_rank(scores, returns)
        mrr_values.append(mrr)

    return np.mean(mrr_values)


class MetricTracker:
    """Tracks and aggregates metrics across training epochs."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.labels = []
        self.ranking_scores = []
        self.actual_returns = []
        self.losses = []
        self.move_losses = []
        self.rank_losses = []

        # Per-day tracking for DAMRR
        self.daily_scores = []
        self.daily_returns = []

    def update(
        self,
        movement_logits: torch.Tensor,
        movement_labels: torch.Tensor,
        ranking_scores: torch.Tensor,
        actual_returns: torch.Tensor,
        loss: float = None,
        move_loss: float = None,
        rank_loss: float = None,
    ):
        """Update metrics with a single snapshot's results."""
        with torch.no_grad():
            preds = movement_logits.argmax(dim=-1).cpu().numpy()
            labels = movement_labels.cpu().numpy()
            scores = ranking_scores.squeeze(-1).cpu().numpy()
            returns = actual_returns.cpu().numpy()

        self.predictions.append(preds)
        self.labels.append(labels)
        self.ranking_scores.append(scores)
        self.actual_returns.append(returns)

        # Per-day for DAMRR
        self.daily_scores.append(scores)
        self.daily_returns.append(returns)

        if loss is not None:
            self.losses.append(loss)
        if move_loss is not None:
            self.move_losses.append(move_loss)
        if rank_loss is not None:
            self.rank_losses.append(rank_loss)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated data."""
        if not self.predictions:
            raise RuntimeError(
                "MetricTracker.compute() called with no data — the dataset has 0 "
                "valid snapshots. This usually means the date types in the samples "
                "dict do not match the keys in the graphs dict. Check dataset.py."
            )
        all_preds = np.concatenate(self.predictions)
        all_labels = np.concatenate(self.labels)

        metrics = {
            "accuracy": accuracy(all_preds, all_labels),
            "precision": precision(all_preds, all_labels),
            "damrr": daily_average_mrr(self.daily_scores, self.daily_returns),
        }

        if self.losses:
            metrics["loss"] = np.mean(self.losses)
        if self.move_losses:
            metrics["move_loss"] = np.mean(self.move_losses)
        if self.rank_losses:
            metrics["rank_loss"] = np.mean(self.rank_losses)

        return metrics
