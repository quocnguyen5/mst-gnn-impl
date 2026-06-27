"""
Vietnam Stock Market Experiment
================================
Runs the full MST-GNN pipeline on Vietnamese stocks (HOSE).

Supports:
    - vn100 : top 100 HOSE stocks (recommended — richer graphs)
    - vn30  : official VN30 index

Usage:
    python -m experiments.run_vn --universe vn100 --aggregator mean
    python -m experiments.run_vn --universe vn30  --aggregator lstm
    python -m experiments.run_vn --universe vn100 --aggregator all

Data period:  2020-01-02 → 2024-06-30 (4.5 years, ~1,100 trading days)
Split:        70% train / 10% val / 20% test  (time-based, no leakage)
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.collector_vn import VietnamStockCollector
from data.preprocessing import StockPreprocessor
from data.graph_builder import GraphBuilder
from data.dataset import DatasetBuilder
from models.mst_gnn import MSTGNN
from train import Trainer, set_seed
from backtest import TradingSimulator
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def _push_results_to_github(dataset: str, save_dir: str):
    """Push checkpoints and logs to GitHub if GITHUB_TOKEN is set."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("  [Git] GITHUB_TOKEN not set — skipping auto-push.", flush=True)
        return
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmds = [
            ["git", "add", "checkpoints/", "logs/"],
            ["git", "commit", "-m", f"Auto: {dataset.upper()} results"],
            ["git", "pull", "--rebase", "--no-edit"],
            ["git", "push"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=120)
            if r.returncode != 0 and "nothing to commit" not in r.stdout:
                logger.warning(f"Git command failed: {' '.join(cmd)}\n{r.stderr}")
        print(f"  [Git] Pushed {dataset.upper()} results to GitHub.", flush=True)
    except Exception as e:
        print(f"  [Git] Push failed (non-fatal): {e}", flush=True)


# ---------------------------------------------------------------------------
# Config preset for Vietnam
# ---------------------------------------------------------------------------

def make_vn_config(universe: str = "vn100", aggregator: str = "mean") -> Config:
    """Build a Config object tuned for the Vietnam market."""
    config = Config()

    # Data
    config.data.dataset = universe
    config.data.raw_data_dir = "data/raw_vn"
    config.data.processed_data_dir = "data/processed_vn"
    config.data.cache_dir = "data/cache_vn"

    # Vietnam date range: 2020-2024
    # (reuse csi300_start/end fields as generic start/end)
    config.data.csi300_start = "2020-01-02"
    config.data.csi300_end = "2024-06-30"

    # Same train/val/test split ratios
    config.data.train_ratio = 0.7
    config.data.val_ratio = 0.1
    config.data.test_ratio = 0.2

    # Feature / graph settings unchanged
    config.data.lookback_window = 5
    config.data.comovement_window = 20
    config.data.comovement_threshold = 0.3

    # Model — same architecture, slightly smaller for VN30
    config.model.stna_aggregator = aggregator

    # Training
    config.train.num_epochs = 200
    config.train.patience = 20
    config.train.experiment_name = f"mst_gnn_{universe}_{aggregator}"
    config.train.save_dir = "checkpoints"

    return config


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_vn_experiment(
    universe: str = "vn100",
    aggregator: str = "mean",
    start_date: str = "2020-01-02",
    end_date: str = "2026-06-26",
):
    """
    Run the complete MST-GNN experiment on Vietnamese stock data.

    Args:
        universe:   "vn30" or "vn100"
        aggregator: "mean", "lstm", or "maxpool"
        start_date: Data start date  "YYYY-MM-DD"
        end_date:   Data end date    "YYYY-MM-DD"
    """
    config = make_vn_config(universe, aggregator)
    set_seed(config.train.seed)
    setup_logger(name="mst_gnn_vn", log_dir="logs")

    import time as _time
    t_total = _time.time()

    logger.info("=" * 60)
    logger.info(f"MST-GNN Vietnam | Universe: {universe.upper()} | Aggregator: {aggregator}")
    logger.info(f"Period: {start_date}  →  {end_date}")
    logger.info("=" * 60)
    print(f"\n{'='*60}", flush=True)
    print(f"  MST-GNN Vietnam: {universe.upper()} | {aggregator}", flush=True)
    print(f"  Period: {start_date} → {end_date}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ---- Phase 1: Data Collection ----------------------------------------
    t1 = _time.time()
    print("[VN Phase 1] Collecting data (loading from cache if available)...", flush=True)
    logger.info("Phase 1: Collecting Vietnam stock data...")
    collector = VietnamStockCollector(
        cache_dir=config.data.raw_data_dir,
        source="VCI",
    )
    try:
        raw_data = collector.collect_all(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        logger.info("Make sure vnstock is installed: pip install vnstock")
        raise

    print(f"  [VN Phase 1 done] {_time.time()-t1:.1f}s — "
          f"{len(raw_data['daily_prices']):,} price rows, "
          f"{len(raw_data['industry'])} industry records", flush=True)
    logger.info(
        f"  Stocks:      {len(raw_data['constituents'])} | "
        f"Price rows: {len(raw_data['daily_prices']):,}"
    )

    # ---- Phase 2: Preprocessing ------------------------------------------
    t2 = _time.time()
    print("\n[VN Phase 2] Preprocessing — computing 13 features & sliding windows...", flush=True)
    logger.info("Phase 2: Preprocessing data...")
    preprocessor = StockPreprocessor(
        lookback_window=config.data.lookback_window,
        prediction_horizon=config.data.prediction_horizon,
    )
    processed_df, samples, trading_dates = preprocessor.process_pipeline(
        raw_data["daily_prices"]
    )
    print(f"  [VN Phase 2 done] {_time.time()-t2:.1f}s — "
          f"{len(trading_dates)} trading days, {len(samples):,} samples", flush=True)

    active_stocks = {
        date: preprocessor.get_active_stocks(processed_df, date)
        for date in trading_dates
    }

    # ---- Phase 3: Graph Construction -------------------------------------
    t3 = _time.time()
    print(f"\n[VN Phase 3] Building graphs for {len(trading_dates)} trading days...", flush=True)
    logger.info("Phase 3: Building multilayer graphs...")

    # Graph cache: skip rebuild if cached pickle exists
    graph_cache_path = os.path.join(
        config.data.processed_data_dir,
        f"graphs_{universe}_{len(trading_dates)}days.pkl",
    )
    os.makedirs(config.data.processed_data_dir, exist_ok=True)

    if os.path.exists(graph_cache_path):
        print(f"  [Graph Cache] Loading from {graph_cache_path} (instant)...", flush=True)
        with open(graph_cache_path, "rb") as f:
            graphs = pickle.load(f)
        logger.info(f"Loaded {len(graphs)} graphs from cache.")
    else:
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
        # Save to cache for next run
        with open(graph_cache_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"  [Graph Cache] Saved to {graph_cache_path}", flush=True)

    print(f"  [VN Phase 3 done] {len(graphs)} graphs in {(_time.time()-t3)/60:.1f} min", flush=True)

    # ---- Phase 4: Dataset ------------------------------------------------
    t4 = _time.time()
    print("\n[VN Phase 4] Creating dataset — matching samples to graphs...", flush=True)
    logger.info("Phase 4: Creating dataset...")
    dataset_builder = DatasetBuilder(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        cache_dir=config.data.processed_data_dir,
    )
    snapshots = dataset_builder.build_snapshots(trading_dates, samples, graphs)
    dataset_builder.save_dataset(snapshots, filename=f"{universe}_snapshots.pkl")
    train_ds, val_ds, test_ds = dataset_builder.split_dataset(snapshots)
    print(f"  [VN Phase 4 done] {_time.time()-t4:.1f}s — "
          f"{len(snapshots)} snapshots | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    # ---- Phase 5: Training -----------------------------------------------
    t5 = _time.time()
    print(f"\n[VN Phase 5] Training MST-GNN ({universe.upper()})...", flush=True)
    logger.info("Phase 5: Training MST-GNN...")
    model = MSTGNN.from_config(config)
    print(f"  Model parameters: {model.count_parameters():,}", flush=True)
    logger.info(f"  Parameters: {model.count_parameters():,}")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )
    test_metrics = trainer.train()
    print(f"  [VN Phase 5 done] {(_time.time()-t5)/60:.1f} min", flush=True)

    # ---- Phase 6: Backtest -----------------------------------------------
    print("\n[VN Phase 6] Trading simulation...", flush=True)
    logger.info("Phase 6: Trading simulation...")
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
            f"cumulative_returns_{universe}.png",
        ),
    )

    # ---- Summary ---------------------------------------------------------
    total_min = (_time.time() - t_total) / 60
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE — Vietnam Stock Market")
    logger.info(f"Universe:       {universe.upper()} ({len(raw_data['constituents'])} stocks)")
    logger.info(f"Period:         {start_date}  →  {end_date}")
    logger.info(f"Aggregator:     {aggregator}")
    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test DAMRR:     {test_metrics['damrr']:.4f}")
    logger.info(f"Total time:     {total_min:.1f} min")
    logger.info("=" * 60)

    print(f"\n{'='*60}", flush=True)
    print(f"  {universe.upper()} DONE in {total_min:.1f} min", flush=True)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}  |  "
          f"Precision: {test_metrics['precision']:.4f}  |  "
          f"DAMRR: {test_metrics['damrr']:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ---- Auto-push to GitHub ----
    _push_results_to_github(universe, config.train.save_dir)

    return test_metrics


def run_all_aggregators(universe: str = "vn100"):
    """Run experiment with all 3 aggregator variants."""
    results = {}
    for agg in ["mean", "lstm", "maxpool"]:
        logger.info(f"\n{'#' * 60}\n# {universe.upper()} — {agg} aggregator\n{'#' * 60}\n")
        results[f"MST-GNN-{agg}"] = run_vn_experiment(universe, agg)

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATOR COMPARISON — Vietnam")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(
            f"  {name:20s} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"DAMRR: {metrics['damrr']:.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MST-GNN on Vietnam stock market (HOSE)"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="vn100",
        choices=["vn30", "vn100"],
        help="Stock universe (default: vn100)",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "lstm", "maxpool", "all"],
        help="STNA aggregator type (default: mean)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-02",
        help="Start date YYYY-MM-DD (default: 2020-01-02)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-06-26",
        help="End date YYYY-MM-DD (default: 2026-06-26)",
    )
    args = parser.parse_args()

    if args.aggregator == "all":
        run_all_aggregators(args.universe)
    else:
        run_vn_experiment(
            universe=args.universe,
            aggregator=args.aggregator,
            start_date=args.start,
            end_date=args.end,
        )
