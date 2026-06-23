"""
Sanity Check Script
====================
Runs a quick check on synthetic data to verify that all modules are working
without shape mismatches or PyTorch execution runtime errors, and that gradient backpropagation flows properly.

Usage:
    python -m experiments.run_sanity_check
"""

import os
import sys
import logging
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import MultilayerTemporalGraphSnapshot, StockTemporalDataset
from models.mst_gnn import MSTGNN
from train import Trainer, set_seed

def create_synthetic_dataset(num_snapshots=5, num_stocks=6, T=5, d=13):
    snapshots = []
    network_names = ["shareholding", "industry", "topicality", "comovement"]
    
    for i in range(num_snapshots):
        date = f"2026060{i+1}"
        stock_codes = [f"stock_{j}" for j in range(num_stocks)]
        node_features = torch.randn(num_stocks, T, d)
        
        networks = {}
        for name in network_names:
            # Let's create random edges between stocks (dense-ish)
            src = []
            dst = []
            for u in range(num_stocks):
                for v in range(num_stocks):
                    if u != v: # no self loops here, they will be added by STNA
                        src.append(u)
                        dst.append(v)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_weight = torch.rand(len(src))
            networks[name] = (edge_index, edge_weight)
            
        movement_labels = torch.randint(0, 2, (num_stocks,), dtype=torch.long)
        return_labels = torch.randn(num_stocks)
        
        snap = MultilayerTemporalGraphSnapshot(
            date=date,
            stock_codes=stock_codes,
            node_features=node_features,
            networks=networks,
            movement_labels=movement_labels,
            return_labels=return_labels
        )
        snapshots.append(snap)
        
    return StockTemporalDataset(snapshots)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sanity_check")
    logger.info("Starting MST-GNN implementation sanity check...")
    
    # 1. Setup config for CPU execution and minimal epochs
    config = Config()
    config.train.device = "cpu"
    config.train.num_epochs = 3
    config.train.patience = 2
    config.train.log_interval = 1
    config.train.experiment_name = "sanity_check_run"
    config.train.save_dir = "checkpoints_sanity"
    
    set_seed(42)
    
    # 2. Build synthetic datasets
    logger.info("Building synthetic datasets...")
    train_ds = create_synthetic_dataset(num_snapshots=5, num_stocks=6, T=5, d=13)
    val_ds = create_synthetic_dataset(num_snapshots=2, num_stocks=6, T=5, d=13)
    test_ds = create_synthetic_dataset(num_snapshots=2, num_stocks=6, T=5, d=13)
    
    # 3. Initialize model
    logger.info("Initializing MST-GNN model...")
    model = MSTGNN.from_config(config)
    
    # 4. Train model
    logger.info("Starting training loop...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds
    )
    
    logger.info("Running trainer.train()...")
    metrics = trainer.train()
    
    logger.info("Sanity check completed successfully!")
    logger.info(f"Final test metrics: {metrics}")
    
    # Cleanup sanity check directory
    import shutil
    if os.path.exists("checkpoints_sanity"):
        shutil.rmtree("checkpoints_sanity")
    if os.path.exists("runs/sanity_check_run"):
        shutil.rmtree("runs/sanity_check_run")

if __name__ == "__main__":
    main()
