"""
Evaluation Script
==================
Standalone evaluation of trained MST-GNN model.
Computes all metrics from the paper: Accuracy, Precision, DAMRR.
"""

import argparse
import logging
import os

import torch

from config import Config
from data.dataset import DatasetBuilder
from models.mst_gnn import MSTGNN
from utils.metrics import MetricTracker
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def evaluate_model(
    model: MSTGNN,
    dataset,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained MST-GNN model
        dataset: StockTemporalDataset
        device: torch device

    Returns:
        Dict of metrics
    """
    model.eval()
    model.reset_temporal_state()
    tracker = MetricTracker()

    with torch.no_grad():
        for idx in range(len(dataset)):
            snapshot = dataset[idx]
            snapshot = snapshot.to(device)

            outputs = model(
                node_features=snapshot.node_features,
                networks=snapshot.networks,
            )

            tracker.update(
                outputs["movement_logits"],
                snapshot.movement_labels,
                outputs["ranking_scores"],
                snapshot.return_labels,
            )

    return tracker.compute()


def load_and_evaluate(
    checkpoint_path: str,
    config: Config = None,
) -> dict:
    """
    Load a trained model from checkpoint and evaluate.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration (loaded from checkpoint if not provided)

    Returns:
        Test metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if config is None:
        config = checkpoint.get("config", Config())

    # Create model
    model = MSTGNN.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    # Load test dataset
    builder = DatasetBuilder(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    snapshots = builder.load_dataset()
    _, _, test_dataset = builder.split_dataset(snapshots)

    # Evaluate
    metrics = evaluate_model(model, test_dataset, device)

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  DAMRR:     {metrics['damrr']:.4f}")
    logger.info("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MST-GNN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csi300",
        choices=["csi300", "csi500"],
        help="Dataset to evaluate on",
    )
    args = parser.parse_args()

    setup_logger()

    config = Config.for_csi300() if args.dataset == "csi300" else Config.for_csi500()
    load_and_evaluate(args.checkpoint, config)
