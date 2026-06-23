"""
Network Combination Analysis — Fig. 7 Reproduction
====================================================
Tests MST-GNN with different combinations of stock network layers
to show that "combining more types of stock networks tends to have
better prediction performance."

Reference: Section V-E of the paper (Fig. 7)
"""

import argparse
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import DatasetBuilder, MultilayerTemporalGraphSnapshot, StockTemporalDataset
from models.mst_gnn import MSTGNN
from train import Trainer, set_seed
from utils.logger import setup_logger
from utils.visualization import plot_network_analysis

import torch

logger = logging.getLogger(__name__)

NETWORK_NAMES = ["shareholding", "industry", "topicality", "comovement"]
SHORT_NAMES = {"shareholding": "S", "industry": "I", "topicality": "T", "comovement": "C"}


def filter_dataset_networks(
    dataset: StockTemporalDataset,
    keep_networks: list,
) -> StockTemporalDataset:
    """
    Create a new dataset keeping only the specified network layers.
    Other layers are replaced with empty graphs.
    """
    new_snapshots = []
    for snapshot in dataset.snapshots:
        new_networks = {}
        for name in NETWORK_NAMES:
            if name in keep_networks:
                new_networks[name] = snapshot.networks.get(
                    name,
                    (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)),
                )
            else:
                new_networks[name] = (
                    torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0),
                )

        new_snap = MultilayerTemporalGraphSnapshot(
            date=snapshot.date,
            stock_codes=snapshot.stock_codes,
            node_features=snapshot.node_features,
            networks=new_networks,
            movement_labels=snapshot.movement_labels,
            return_labels=snapshot.return_labels,
        )
        new_snapshots.append(new_snap)

    return StockTemporalDataset(new_snapshots)


def run_network_analysis(dataset: str = "csi300"):
    """
    Test all meaningful network layer combinations.
    """
    config = Config.for_csi300() if dataset == "csi300" else Config.for_csi500()
    set_seed(config.train.seed)
    setup_logger(name="mst_gnn", log_dir="logs")

    # Load cached dataset
    builder = DatasetBuilder(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        cache_dir=config.data.processed_data_dir,
    )

    try:
        snapshots = builder.load_dataset(filename=f"{dataset}_snapshots.pkl")
    except FileNotFoundError:
        logger.error("Dataset not found. Run main experiment first.")
        return

    train_ds, val_ds, test_ds = builder.split_dataset(snapshots)

    # Generate combinations: single layers, pairs, triples, and full
    combinations = []

    # Singles
    for name in NETWORK_NAMES:
        combinations.append([name])

    # Pairs
    for combo in itertools.combinations(NETWORK_NAMES, 2):
        combinations.append(list(combo))

    # Triples
    for combo in itertools.combinations(NETWORK_NAMES, 3):
        combinations.append(list(combo))

    # Full
    combinations.append(NETWORK_NAMES)

    results = {}
    for combo in combinations:
        combo_name = "+".join(SHORT_NAMES[n] for n in combo)
        logger.info(f"\n--- Network Combination: {combo_name} ({combo}) ---")

        # Filter datasets to keep only selected networks
        filtered_train = filter_dataset_networks(train_ds, combo)
        filtered_val = filter_dataset_networks(val_ds, combo)
        filtered_test = filter_dataset_networks(test_ds, combo)

        # Train with this combination
        config.train.experiment_name = f"network_{dataset}_{combo_name}"
        config.train.num_epochs = 100  # Reduced for speed
        config.train.patience = 15

        model = MSTGNN.from_config(config)
        trainer = Trainer(
            model, config, filtered_train, filtered_val, filtered_test
        )
        metrics = trainer.train()
        results[combo_name] = metrics

    # Plot results
    plot_network_analysis(
        results,
        save_path=os.path.join(
            config.train.save_dir, f"network_analysis_{dataset}.png"
        ),
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("NETWORK COMBINATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(
        f"{'Combination':15s} | {'Accuracy':>8s} | {'Precision':>9s} | {'DAMRR':>8s}"
    )
    logger.info("-" * 70)
    for name, metrics in sorted(
        results.items(), key=lambda x: len(x[0])
    ):
        logger.info(
            f"{name:15s} | "
            f"{metrics['accuracy']:8.4f} | "
            f"{metrics['precision']:9.4f} | "
            f"{metrics['damrr']:8.4f}"
        )
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run network combination analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csi300",
        choices=["csi300", "csi500"],
    )
    args = parser.parse_args()

    run_network_analysis(args.dataset)
