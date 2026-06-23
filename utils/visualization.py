"""
Visualization Utilities
========================
Plotting functions for training curves, evaluation results,
and trading simulation outcomes.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_training_curves(
    train_metrics: List[Dict],
    val_metrics: List[Dict],
    save_path: str = None,
):
    """
    Plot training and validation loss/metric curves.

    Args:
        train_metrics: List of metric dicts per epoch
        val_metrics: List of metric dicts per epoch
        save_path: If provided, save figure to this path
    """
    epochs = range(1, len(train_metrics) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MST-GNN Training Progress", fontsize=16, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [m["loss"] for m in train_metrics], label="Train", linewidth=2)
    ax.plot(epochs, [m["loss"] for m in val_metrics], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [m["accuracy"] for m in train_metrics], label="Train", linewidth=2)
    ax.plot(epochs, [m["accuracy"] for m in val_metrics], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Movement Accuracy")
    ax.legend()

    # Precision
    ax = axes[1, 0]
    ax.plot(epochs, [m["precision"] for m in train_metrics], label="Train", linewidth=2)
    ax.plot(epochs, [m["precision"] for m in val_metrics], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Movement Precision")
    ax.legend()

    # DAMRR
    ax = axes[1, 1]
    ax.plot(epochs, [m["damrr"] for m in train_metrics], label="Train", linewidth=2)
    ax.plot(epochs, [m["damrr"] for m in val_metrics], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DAMRR")
    ax.set_title("Daily Average MRR")
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: str = None,
):
    """
    Plot a comparison table of different methods' results.

    Args:
        results: Dict mapping method_name -> {metric: value}
        metrics: List of metric names to include
        save_path: If provided, save figure
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "damrr"]

    methods = list(results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in methods]
        bars = ax.barh(methods, values, color=colors)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_returns(
    returns_dict: Dict[str, List[float]],
    dates: List = None,
    title: str = "Cumulative Returns",
    save_path: str = None,
):
    """
    Plot cumulative return curves for trading simulation.

    Args:
        returns_dict: Dict mapping strategy_name -> daily return list
        dates: Trading dates
        title: Plot title
        save_path: If provided, save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, returns in returns_dict.items():
        cum_returns = np.cumprod(1 + np.array(returns))
        if dates is not None and len(dates) == len(returns):
            ax.plot(dates, cum_returns, label=name, linewidth=2)
        else:
            ax.plot(cum_returns, label=name, linewidth=2)

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    if dates is not None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_results(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: str = None,
):
    """
    Plot ablation study results as grouped bar chart.

    Args:
        ablation_results: Dict mapping variant_name -> {metric: value}
        save_path: If provided, save figure
    """
    variants = list(ablation_results.keys())
    metrics = ["accuracy", "precision", "damrr"]

    x = np.arange(len(variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [ablation_results[v].get(metric, 0) for v in variants]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper())

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_network_analysis(
    network_combinations: Dict[str, Dict[str, float]],
    save_path: str = None,
):
    """
    Plot performance vs number of network layers (Fig. 7 reproduction).

    Args:
        network_combinations: Dict mapping combination_name -> {metric: value}
        save_path: If provided, save figure
    """
    combos = list(network_combinations.keys())
    metrics = ["accuracy", "precision", "damrr"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Performance vs Network Layer Combinations",
        fontsize=14,
        fontweight="bold",
    )

    for ax, metric in zip(axes, metrics):
        values = [network_combinations[c].get(metric, 0) for c in combos]
        ax.bar(range(len(combos)), values, color="steelblue", alpha=0.8)
        ax.set_xlabel("Network Combination")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
