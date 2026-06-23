"""
Ablation Study — Table VI Reproduction
========================================
Tests the contribution of each module by removing them one at a time:

1. MST-GNN (full model)
2. w/o STNA — remove spatial-temporal neighborhood aggregation
3. w/o HOFF — remove cross-layer high-order feature fusion
4. w/o Multitask — use only movement loss (δ=1)
5. w/o Temporal — remove temporal connections in STNA
6. Simple concat — replace HOFF with simple concatenation

Reference: Section V-D (Table VI)

"The original MST-GNN model cannot lead to the best results in all
metrics. However, it has the lowest average rank score."
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from data.dataset import DatasetBuilder
from models.mst_gnn import MSTGNN
from models.feature_encoder import AttentiveLSTMEncoder
from models.stna import MultiLayerSTNA
from models.hoff import HOFF
from models.predictor import MultitaskPredictor, MultitaskLoss
from train import Trainer, set_seed
from utils.logger import setup_logger
from utils.visualization import plot_ablation_results
from utils.metrics import MetricTracker

logger = logging.getLogger(__name__)


class MSTGNNWithoutSTNA(MSTGNN):
    """MST-GNN variant without STNA module.
    
    Directly feeds encoded features to HOFF without spatial-temporal 
    aggregation (uses same representation for all network layers).
    """

    def forward(self, node_features, networks, return_intermediate=False):
        encoded = self.encoder(node_features)

        # Skip STNA — use encoded features directly for all layers
        network_names = self.NETWORK_NAMES[: self.num_networks]
        stna_outputs = {name: encoded for name in network_names}

        fused = self.hoff(stna_outputs, network_names)
        movement_logits, ranking_scores = self.predictor(fused)

        return {"movement_logits": movement_logits, "ranking_scores": ranking_scores}


class MSTGNNWithoutHOFF(MSTGNN):
    """MST-GNN variant without HOFF — uses simple concatenation + MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace HOFF with simple concatenation + linear
        concat_dim = self.num_networks * kwargs.get("stna_hidden_dim", 64)
        output_dim = kwargs.get("hoff_output_dim", 64)
        self.simple_fusion = torch.nn.Sequential(
            torch.nn.Linear(concat_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )

    def forward(self, node_features, networks, return_intermediate=False):
        encoded = self.encoder(node_features)
        network_names = self.NETWORK_NAMES[: self.num_networks]
        stna_outputs = self.stna(encoded, networks, network_names=network_names)

        # Simple concatenation instead of HOFF
        representations = [stna_outputs[name] for name in network_names]
        concat = torch.cat(representations, dim=-1)
        fused = self.simple_fusion(concat)

        movement_logits, ranking_scores = self.predictor(fused)
        return {"movement_logits": movement_logits, "ranking_scores": ranking_scores}


def run_ablation(dataset: str = "csi300"):
    """
    Run ablation study with all variants.

    Variants:
    1. Full MST-GNN
    2. w/o STNA
    3. w/o HOFF (simple concat)
    4. w/o Ranking (δ=1, movement only)
    5. w/o Movement (δ=0, ranking only)
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
        logger.error(
            "Dataset not found. Please run the main experiment first: "
            "python -m experiments.run_main"
        )
        return

    train_ds, val_ds, test_ds = builder.split_dataset(snapshots)

    results = {}

    # --- Variant 1: Full MST-GNN ---
    logger.info("\n--- Ablation: Full MST-GNN ---")
    config.train.experiment_name = f"ablation_{dataset}_full"
    model = MSTGNN.from_config(config)
    trainer = Trainer(model, config, train_ds, val_ds, test_ds)
    results["Full MST-GNN"] = trainer.train()

    # --- Variant 2: w/o STNA ---
    logger.info("\n--- Ablation: w/o STNA ---")
    config.train.experiment_name = f"ablation_{dataset}_no_stna"
    model = MSTGNNWithoutSTNA.from_config(config)
    trainer = Trainer(model, config, train_ds, val_ds, test_ds)
    results["w/o STNA"] = trainer.train()

    # --- Variant 3: w/o HOFF (simple concat) ---
    logger.info("\n--- Ablation: w/o HOFF ---")
    config.train.experiment_name = f"ablation_{dataset}_no_hoff"
    model = MSTGNNWithoutHOFF.from_config(config)
    trainer = Trainer(model, config, train_ds, val_ds, test_ds)
    results["w/o HOFF"] = trainer.train()

    # --- Variant 4: Movement only (δ=1) ---
    logger.info("\n--- Ablation: Movement only ---")
    config_move = Config.for_csi300() if dataset == "csi300" else Config.for_csi500()
    config_move.train.delta = 1.0
    config_move.train.experiment_name = f"ablation_{dataset}_move_only"
    model = MSTGNN.from_config(config_move)
    trainer = Trainer(model, config_move, train_ds, val_ds, test_ds)
    results["Movement Only (δ=1)"] = trainer.train()

    # --- Variant 5: Ranking only (δ=0) ---
    logger.info("\n--- Ablation: Ranking only ---")
    config_rank = Config.for_csi300() if dataset == "csi300" else Config.for_csi500()
    config_rank.train.delta = 0.0
    config_rank.train.experiment_name = f"ablation_{dataset}_rank_only"
    model = MSTGNN.from_config(config_rank)
    trainer = Trainer(model, config_rank, train_ds, val_ds, test_ds)
    results["Ranking Only (δ=0)"] = trainer.train()

    # Plot results
    plot_ablation_results(
        results,
        save_path=os.path.join(
            config.train.save_dir, f"ablation_{dataset}.png"
        ),
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"{'Variant':25s} | {'Accuracy':>8s} | {'Precision':>9s} | {'DAMRR':>8s}"
    )
    logger.info("-" * 70)
    for name, metrics in results.items():
        logger.info(
            f"{name:25s} | "
            f"{metrics['accuracy']:8.4f} | "
            f"{metrics['precision']:9.4f} | "
            f"{metrics['damrr']:8.4f}"
        )
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csi300",
        choices=["csi300", "csi500"],
    )
    args = parser.parse_args()

    run_ablation(args.dataset)
