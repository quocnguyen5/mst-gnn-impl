"""
Main Experiment — Tables IV & V Reproduction
=============================================
Runs the full MST-GNN training and evaluation pipeline.

Reproduces the main results from Tables IV (CSI 300) and V (CSI 500).

Usage:
    python -m experiments.run_main --dataset csi300
    python -m experiments.run_main --dataset csi500
"""

import argparse
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.collector import StockDataCollector
from data.preprocessing import StockPreprocessor
from data.graph_builder import GraphBuilder
from data.dataset import DatasetBuilder
from models.mst_gnn import MSTGNN
from train import Trainer, set_seed
from backtest import TradingSimulator
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def run_experiment(dataset: str = "csi300", aggregator: str = "mean"):
    """
    Run the complete MST-GNN experiment.

    Args:
        dataset: "csi300" or "csi500"
        aggregator: "mean", "lstm", or "maxpool"
    """
    # Configuration
    config = Config.for_csi300() if dataset == "csi300" else Config.for_csi500()
    config.model.stna_aggregator = aggregator
    config.train.experiment_name = f"mst_gnn_{dataset}_{aggregator}"

    set_seed(config.train.seed)
    setup_logger(name="mst_gnn", log_dir="logs")

    logger.info("=" * 60)
    logger.info(f"MST-GNN Experiment: {dataset.upper()} with {aggregator} aggregator")
    logger.info("=" * 60)

    # ---- Phase 1: Data Collection ----
    t1 = time.time()
    print("\n[Phase 1] Collecting data (loading from cache if available)...", flush=True)
    logger.info("Phase 1: Collecting data...")
    collector = StockDataCollector(cache_dir=config.data.raw_data_dir)

    try:
        raw_data = collector.collect_all(
            dataset=dataset,
            start_date=(
                config.data.csi300_start
                if dataset == "csi300"
                else config.data.csi500_start
            ),
            end_date=(
                config.data.csi300_end
                if dataset == "csi300"
                else config.data.csi500_end
            ),
        )
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        logger.info("Please ensure you have internet access and AKShare is installed.")
        raise
    print(f"  [Phase 1 done] {time.time()-t1:.1f}s — "
          f"{len(raw_data['daily_prices']):,} price rows, "
          f"{len(raw_data['industry'])} industry records", flush=True)

    # ---- Phase 2: Preprocessing ----
    t2 = time.time()
    print("\n[Phase 2] Preprocessing — computing 13 features & sliding windows...", flush=True)
    logger.info("Phase 2: Preprocessing data...")
    preprocessor = StockPreprocessor(
        lookback_window=config.data.lookback_window,
        prediction_horizon=config.data.prediction_horizon,
    )
    processed_df, samples, trading_dates = preprocessor.process_pipeline(
        raw_data["daily_prices"]
    )

    # Get active stocks per date
    active_stocks = {}
    for date in trading_dates:
        active_stocks[date] = preprocessor.get_active_stocks(processed_df, date)
    print(f"  [Phase 2 done] {time.time()-t2:.1f}s — "
          f"{len(trading_dates)} trading days, {len(samples):,} samples", flush=True)

    # ---- Phase 3: Graph Construction ----
    t3 = time.time()
    logger.info("Phase 3: Building multilayer graphs...")
    graph_builder = GraphBuilder(
        comovement_window=config.data.comovement_window,
        comovement_threshold=config.data.comovement_threshold,
        num_topics=config.data.num_topics,
        topic_similarity_threshold=config.data.topic_similarity_threshold,
    )

    graphs = graph_builder.build_temporal_multilayer_graphs(
        trading_dates=trading_dates,
        price_df=processed_df,
        industry_df=raw_data["industry"],
        shareholding_df=raw_data["shareholding"],
        news_df=raw_data["news"],
        active_stocks_per_date=active_stocks,
    )

    # ---- Phase 4: Dataset Creation ----
    t4 = time.time()
    print("\n[Phase 4] Creating dataset — matching samples to graphs...", flush=True)
    logger.info("Phase 4: Creating dataset...")
    dataset_builder = DatasetBuilder(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        cache_dir=config.data.processed_data_dir,
    )

    snapshots = dataset_builder.build_snapshots(trading_dates, samples, graphs)
    dataset_builder.save_dataset(snapshots, filename=f"{dataset}_snapshots.pkl")

    train_ds, val_ds, test_ds = dataset_builder.split_dataset(snapshots)
    print(f"  [Phase 4 done] {time.time()-t4:.1f}s — "
          f"{len(snapshots)} snapshots | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}",
          flush=True)

    # ---- Phase 5: Training ----
    t5 = time.time()
    print("\n[Phase 5] Training MST-GNN...", flush=True)
    logger.info("Phase 5: Training MST-GNN...")
    model = MSTGNN.from_config(config)
    print(f"  Model parameters: {model.count_parameters():,}", flush=True)
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )

    test_metrics = trainer.train()

    # ---- Phase 6: Trading Simulation ----
    logger.info("Phase 6: Running trading simulation...")
    dates, codes, preds, scores, returns = trainer.get_predictions(test_ds)

    simulator = TradingSimulator(
        top_k_stocks=config.backtest.top_k_stocks,
        transaction_cost=config.backtest.transaction_cost,
    )

    backtest_results = simulator.simulate(dates, codes, scores, returns)
    report = simulator.generate_report(backtest_results)

    logger.info("\n" + report.to_string(index=False))

    simulator.plot_results(
        backtest_results,
        dates=dates,
        save_path=os.path.join(
            config.train.save_dir,
            f"cumulative_returns_{dataset}.png",
        ),
    )

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Dataset: {dataset.upper()}")
    logger.info(f"Aggregator: {aggregator}")
    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test DAMRR:     {test_metrics['damrr']:.4f}")
    logger.info("=" * 60)

    return test_metrics


def run_all_aggregators(dataset: str = "csi300"):
    """Run experiment with all 3 aggregator types (MST-GNN variants)."""
    results = {}
    for agg in ["mean", "lstm", "maxpool"]:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Running with {agg} aggregator")
        logger.info(f"{'#' * 60}\n")
        results[f"MST-GNN-{agg}"] = run_experiment(dataset, agg)

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATOR COMPARISON")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(
            f"  {name:20s} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"DAMRR: {metrics['damrr']:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MST-GNN main experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="csi300",
        choices=["csi300", "csi500"],
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "lstm", "maxpool", "all"],
    )
    args = parser.parse_args()

    if args.aggregator == "all":
        run_all_aggregators(args.dataset)
    else:
        run_experiment(args.dataset, args.aggregator)
