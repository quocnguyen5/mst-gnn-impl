"""
MST-GNN Configuration
=====================
Central configuration for all hyperparameters and settings.
Based on: "Graph Representation Learning of Multilayer Spatial-Temporal
Networks for Stock Predictions" (IEEE TCSS, 2024)

All equation references (Eq. X) refer to the original paper.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data collection and preprocessing configuration."""

    # --- Dataset Selection ---
    dataset: str = "csi300"  # "csi300" or "csi500"

    # --- Time Periods (Table I in paper) ---
    # CSI 300: original paper used 2018-01-02 to 2022-06-30
    # CSI 500: original paper used 2019-01-02 to 2022-06-30
    # Extended: fetching up to 2026-06-26 for richer test coverage
    csi300_start: str = "2018-01-02"
    csi300_end: str = "2026-06-26"
    csi500_start: str = "2018-01-02"
    csi500_end: str = "2026-06-26"

    # --- Train/Val/Test Split Ratios (time-based) ---
    # Paper: roughly 70% train, 10% validation, 20% test
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # --- Stock Features ---
    # 5 raw features + 8 derivative features = 13 total (Section IV-A)
    raw_features: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    # Derivative features computed in preprocessing
    # 1. daily price change
    # 2. percentage of daily price change
    # 3-5. 5/10/20-day MA of close price
    # 6-8. 5/10/20-day MA of volume
    num_features: int = 13  # d in the paper

    # --- Historical Window ---
    lookback_window: int = 5  # T in paper: number of historical trading days
    prediction_horizon: int = 1  # s in paper: predict s days ahead

    # --- Comovement Network ---
    comovement_window: int = 20  # rolling window for correlation
    comovement_threshold: float = 0.3  # threshold for edge creation

    # --- Topicality Network ---
    num_topics: int = 50  # number of LDA topics
    topic_similarity_threshold: float = 0.2  # threshold for topicality edges

    # --- Data paths ---
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"


@dataclass
class ModelConfig:
    """MST-GNN model architecture configuration."""

    # --- Stock Feature Encoding (Module A, Eqs. 4-7) ---
    input_dim: int = 13  # d: number of stock features
    lstm_hidden_dim: int = 64  # d1: LSTM hidden state dimension
    lstm_num_layers: int = 1  # number of LSTM layers

    # --- Spatial-Temporal Neighborhood Aggregation (Module B, Eqs. 8-12) ---
    stna_depth: int = 2  # K: aggregation depth
    stna_aggregator: str = "mean"  # "mean", "lstm", or "maxpool"
    stna_hidden_dim: int = 64  # hidden dimension in STNA

    # --- Cross-Layer High-Order Feature Fusion (Module C, Eqs. 13-15) ---
    num_network_layers: int = 4  # M: number of stock network types
    cross_network_layers: int = 3  # C: number of cross network layers
    deep_network_layers: int = 2  # number of MLP layers in deep network
    deep_network_dim: int = 128  # hidden dim of deep network

    # --- Multitask Prediction (Module D, Eqs. 17-19) ---
    prediction_hidden_dim: int = 64  # hidden dim in prediction heads
    num_classes: int = 2  # binary movement: up/down

    # --- Dropout ---
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Training configuration."""

    # --- Optimization ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5  # c in Eq. 17: L2 regularization
    batch_size: int = 1  # temporal graphs are processed one snapshot at a time
    num_epochs: int = 200
    patience: int = 20  # early stopping patience

    # --- Multitask Loss (Eq. 17) ---
    # L = delta * L_move + (1 - delta) * L_rank + c * ||Theta||^2
    delta: float = 0.5  # task weight balance
    margin: float = 0.1  # margin for pairwise ranking loss

    # --- Gradient Clipping ---
    max_grad_norm: float = 1.0

    # --- Learning Rate Scheduling ---
    lr_scheduler: str = "cosine"  # "cosine", "step", or "none"
    lr_step_size: int = 50
    lr_gamma: float = 0.5

    # --- Device ---
    device: str = "cuda"  # "cuda" or "cpu" (Colab T4)

    # --- Reproducibility ---
    seed: int = 42

    # --- Logging ---
    log_interval: int = 10  # log every N epochs
    save_dir: str = "checkpoints"
    experiment_name: str = "mst_gnn"


@dataclass
class BacktestConfig:
    """Trading simulation configuration."""

    top_k_stocks: List[int] = field(default_factory=lambda: [5, 10])
    transaction_cost: float = 0.0003  # 0.03% as in the paper
    initial_capital: float = 1_000_000.0


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self):
        """Ensure consistency between configs."""
        self.model.input_dim = self.data.num_features

    @classmethod
    def for_csi300(cls) -> "Config":
        """Preset for CSI 300 experiments."""
        cfg = cls()
        cfg.data.dataset = "csi300"
        return cfg

    @classmethod
    def for_csi500(cls) -> "Config":
        """Preset for CSI 500 experiments."""
        cfg = cls()
        cfg.data.dataset = "csi500"
        return cfg


# Global default config instance
DEFAULT_CONFIG = Config()
