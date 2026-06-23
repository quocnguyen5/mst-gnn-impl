"""
Dataset
=======
PyTorch Dataset for temporal multilayer stock graphs.

Creates temporal graph snapshots where each snapshot at time t contains:
- Node features: x_{i,t} ∈ R^{T×d} for each stock i (sliding window)
- Edge indices and weights for each of the 4 network layers
- Labels: movement direction (binary) and return ratio

Handles time-based train/val/test splitting to prevent data leakage.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultilayerTemporalGraphSnapshot:
    """
    A single temporal graph snapshot at time t.

    Represents the multilayer graph Gₜ = {Gₜ,q, q ∈ R} with:
    - Node features for all active stocks
    - Edge structure for each network layer
    - Movement and return labels
    """

    def __init__(
        self,
        date,
        stock_codes: List[str],
        node_features: torch.Tensor,  # (num_stocks, T, d)
        networks: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        movement_labels: torch.Tensor,  # (num_stocks,)
        return_labels: torch.Tensor,  # (num_stocks,)
    ):
        self.date = date
        self.stock_codes = stock_codes
        self.num_stocks = len(stock_codes)
        self.node_features = node_features
        self.networks = networks  # {name: (edge_index, edge_weight)}
        self.movement_labels = movement_labels
        self.return_labels = return_labels

    def to(self, device: torch.device) -> "MultilayerTemporalGraphSnapshot":
        """Move all tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        self.movement_labels = self.movement_labels.to(device)
        self.return_labels = self.return_labels.to(device)
        self.networks = {
            name: (ei.to(device), ew.to(device))
            for name, (ei, ew) in self.networks.items()
        }
        return self


class StockTemporalDataset(Dataset):
    """
    Dataset of temporal multilayer graph snapshots.

    Each item is a MultilayerTemporalGraphSnapshot for a single trading day.
    """

    def __init__(
        self,
        snapshots: List[MultilayerTemporalGraphSnapshot],
    ):
        self.snapshots = snapshots

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> MultilayerTemporalGraphSnapshot:
        return self.snapshots[idx]


class DatasetBuilder:
    """
    Builds the full temporal dataset from preprocessed data and graphs.
    Handles train/val/test splitting.
    """

    NETWORK_NAMES = ["shareholding", "industry", "topicality", "comovement"]

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        cache_dir: str = "data/processed",
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def build_snapshots(
        self,
        trading_dates: List,
        samples: Dict,
        graphs: Dict,
    ) -> List[MultilayerTemporalGraphSnapshot]:
        """
        Build temporal graph snapshots from preprocessed samples and graphs.

        Args:
            trading_dates: Sorted list of trading dates
            samples: Dict mapping (stock_code, date) -> {features, movement, return}
            graphs: Dict mapping date -> {stock_codes, networks}

        Returns:
            List of MultilayerTemporalGraphSnapshot objects
        """
        snapshots = []

        for date in trading_dates:
            if date not in graphs:
                continue

            graph_info = graphs[date]
            stock_codes = graph_info["stock_codes"]
            networks = graph_info["networks"]

            # Collect node features and labels for active stocks
            features_list = []
            movements = []
            returns = []
            valid_codes = []

            for code in stock_codes:
                key = (code, date)
                if key in samples:
                    sample = samples[key]
                    features_list.append(sample["features"])
                    movements.append(sample["movement"])
                    returns.append(sample["return"])
                    valid_codes.append(code)

            if len(valid_codes) < 2:
                # Need at least 2 stocks for pairwise ranking
                continue

            # Convert to tensors
            node_features = torch.tensor(
                np.stack(features_list), dtype=torch.float32
            )  # (n, T, d)
            movement_labels = torch.tensor(movements, dtype=torch.long)
            return_labels = torch.tensor(returns, dtype=torch.float32)

            # Re-index networks for valid_codes only
            code_to_new_idx = {code: i for i, code in enumerate(valid_codes)}
            old_to_new = {}
            original_codes = graph_info["stock_codes"]
            for old_idx, code in enumerate(original_codes):
                if code in code_to_new_idx:
                    old_to_new[old_idx] = code_to_new_idx[code]

            torch_networks = {}
            for net_name in self.NETWORK_NAMES:
                if net_name in networks:
                    ei, ew = networks[net_name]
                    if ei.shape[1] > 0:
                        new_edges = []
                        new_weights = []
                        for e_idx in range(ei.shape[1]):
                            src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
                            if src in old_to_new and dst in old_to_new:
                                new_edges.append(
                                    [old_to_new[src], old_to_new[dst]]
                                )
                                new_weights.append(float(ew[e_idx]))

                        if new_edges:
                            torch_networks[net_name] = (
                                torch.tensor(new_edges, dtype=torch.long).t(),
                                torch.tensor(new_weights, dtype=torch.float32),
                            )
                        else:
                            torch_networks[net_name] = (
                                torch.zeros(2, 0, dtype=torch.long),
                                torch.zeros(0, dtype=torch.float32),
                            )
                    else:
                        torch_networks[net_name] = (
                            torch.zeros(2, 0, dtype=torch.long),
                            torch.zeros(0, dtype=torch.float32),
                        )
                else:
                    torch_networks[net_name] = (
                        torch.zeros(2, 0, dtype=torch.long),
                        torch.zeros(0, dtype=torch.float32),
                    )

            snapshot = MultilayerTemporalGraphSnapshot(
                date=date,
                stock_codes=valid_codes,
                node_features=node_features,
                networks=torch_networks,
                movement_labels=movement_labels,
                return_labels=return_labels,
            )
            snapshots.append(snapshot)

        logger.info(f"Built {len(snapshots)} temporal graph snapshots.")
        return snapshots

    def split_dataset(
        self, snapshots: List[MultilayerTemporalGraphSnapshot]
    ) -> Tuple[StockTemporalDataset, StockTemporalDataset, StockTemporalDataset]:
        """
        Time-based train/val/test split (no data leakage).

        The paper states: "We split the data in chronological order to avoid
        information leakage."
        """
        n = len(snapshots)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_snapshots = snapshots[:train_end]
        val_snapshots = snapshots[train_end:val_end]
        test_snapshots = snapshots[val_end:]

        logger.info(
            f"Dataset split — Train: {len(train_snapshots)}, "
            f"Val: {len(val_snapshots)}, Test: {len(test_snapshots)}"
        )

        return (
            StockTemporalDataset(train_snapshots),
            StockTemporalDataset(val_snapshots),
            StockTemporalDataset(test_snapshots),
        )

    def save_dataset(
        self,
        snapshots: List[MultilayerTemporalGraphSnapshot],
        filename: str = "snapshots.pkl",
    ):
        """Save processed snapshots to disk."""
        path = os.path.join(self.cache_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(snapshots, f)
        logger.info(f"Saved {len(snapshots)} snapshots to {path}")

    def load_dataset(
        self, filename: str = "snapshots.pkl"
    ) -> List[MultilayerTemporalGraphSnapshot]:
        """Load processed snapshots from disk."""
        path = os.path.join(self.cache_dir, filename)
        with open(path, "rb") as f:
            snapshots = pickle.load(f)
        logger.info(f"Loaded {len(snapshots)} snapshots from {path}")
        return snapshots


def collate_snapshots(
    batch: List[MultilayerTemporalGraphSnapshot],
) -> MultilayerTemporalGraphSnapshot:
    """
    Custom collate function for DataLoader.
    Since each snapshot has a different number of nodes,
    we process them one at a time (batch_size=1).
    """
    assert len(batch) == 1, "Batch size must be 1 for variable-size graphs."
    return batch[0]
