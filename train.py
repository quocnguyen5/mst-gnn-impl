"""
Training Pipeline
==================
End-to-end training loop for MST-GNN with:
- Multitask loss optimization (Eq. 17)
- Early stopping on validation metrics
- Gradient clipping
- Learning rate scheduling
- Checkpoint saving
- TensorBoard logging

Optimized for Google Colab T4 (free tier).
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from config import Config
from data.dataset import StockTemporalDataset, MultilayerTemporalGraphSnapshot
from models.mst_gnn import MSTGNN
from models.predictor import MultitaskLoss
from utils.metrics import MetricTracker
from utils.logger import setup_logger, setup_tensorboard
from utils.visualization import plot_training_curves

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """MST-GNN training manager."""

    def __init__(
        self,
        model: MSTGNN,
        config: Config,
        train_dataset: StockTemporalDataset,
        val_dataset: StockTemporalDataset,
        test_dataset: Optional[StockTemporalDataset] = None,
    ):
        self.config = config
        self.device = torch.device(
            config.train.device
            if torch.cuda.is_available() and config.train.device == "cuda"
            else "cpu"
        )

        # Model
        self.model = model.to(self.device)
        logger.info(
            f"Model parameters: {model.count_parameters():,}"
        )
        logger.info(f"Module params: {model.get_module_parameters()}")
        logger.info(f"Device: {self.device}")

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Loss function (Eq. 17) — paper default, no class weights
        self.criterion = MultitaskLoss(
            delta=config.train.delta,
            margin=config.train.margin,
        )

        # Optimizer with L2 regularization (c·||Θ||² in Eq. 17)
        self.optimizer = Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # Learning rate scheduler
        if config.train.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.train.num_epochs,
                eta_min=1e-6,
            )
        elif config.train.lr_scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.train.lr_step_size,
                gamma=config.train.lr_gamma,
            )
        else:
            self.scheduler = None

        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_epoch = 1  # for resume support

        # Periodic checkpoint interval
        self.checkpoint_interval = 20  # save every N epochs

        # Directories
        os.makedirs(config.train.save_dir, exist_ok=True)

        # TensorBoard
        self.writer = setup_tensorboard(
            os.path.join("runs", config.train.experiment_name)
        )

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch over all snapshots."""
        self.model.train()
        self.model.reset_temporal_state()
        tracker = MetricTracker()

        for idx in range(len(self.train_dataset)):
            snapshot = self.train_dataset[idx]
            snapshot = snapshot.to(self.device)

            # Forward pass
            outputs = self.model(
                node_features=snapshot.node_features,
                networks=snapshot.networks,
                stock_codes=snapshot.stock_codes,
            )

            # Compute multitask loss (Eq. 17)
            total_loss, move_loss, rank_loss = self.criterion(
                outputs["movement_logits"],
                snapshot.movement_labels,
                outputs["ranking_scores"],
                snapshot.return_labels,
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.max_grad_norm,
            )

            self.optimizer.step()

            # Track metrics
            tracker.update(
                outputs["movement_logits"],
                snapshot.movement_labels,
                outputs["ranking_scores"],
                snapshot.return_labels,
                loss=total_loss.item(),
                move_loss=move_loss.item(),
                rank_loss=rank_loss.item(),
            )

        return tracker.compute()

    @torch.no_grad()
    def _evaluate(self, dataset: StockTemporalDataset) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        self.model.reset_temporal_state()
        tracker = MetricTracker()

        for idx in range(len(dataset)):
            snapshot = dataset[idx]
            snapshot = snapshot.to(self.device)

            outputs = self.model(
                node_features=snapshot.node_features,
                networks=snapshot.networks,
                stock_codes=snapshot.stock_codes,
            )

            total_loss, move_loss, rank_loss = self.criterion(
                outputs["movement_logits"],
                snapshot.movement_labels,
                outputs["ranking_scores"],
                snapshot.return_labels,
            )

            tracker.update(
                outputs["movement_logits"],
                snapshot.movement_labels,
                outputs["ranking_scores"],
                snapshot.return_labels,
                loss=total_loss.item(),
                move_loss=move_loss.item(),
                rank_loss=rank_loss.item(),
            )

        return tracker.compute()

    def resume_from_checkpoint(self) -> bool:
        """
        Automatically detect and load the latest periodic checkpoint.
        Returns True if a checkpoint was loaded, False otherwise.
        """
        import glob as _glob

        exp_name = self.config.train.experiment_name
        pattern = os.path.join(
            self.config.train.save_dir, f"ckpt_{exp_name}_epoch*.pt"
        )
        ckpt_files = sorted(_glob.glob(pattern))
        if not ckpt_files:
            return False

        latest_ckpt = ckpt_files[-1]
        logger.info(f"Found checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        logger.info(
            f"Resumed from epoch {checkpoint['epoch']} "
            f"(best_epoch={self.best_epoch}, "
            f"best_val_loss={self.best_val_loss:.4f}, "
            f"patience={self.patience_counter}/{self.config.train.patience})"
        )
        print(
            f"  [Resume] Loaded checkpoint epoch {checkpoint['epoch']} — "
            f"continuing from epoch {self.start_epoch}",
            flush=True,
        )
        return True

    def _save_periodic_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """Save a periodic checkpoint for crash recovery."""
        exp_name = self.config.train.experiment_name
        path = os.path.join(
            self.config.train.save_dir, f"ckpt_{exp_name}_epoch{epoch:04d}.pt"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "metrics": val_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Periodic checkpoint saved: {path}")

    def train(self, auto_resume: bool = True) -> Dict[str, float]:
        """
        Full training loop with early stopping and checkpoint resume.

        Args:
            auto_resume: If True, automatically resume from the latest
                         periodic checkpoint if one exists.

        Returns:
            Best validation metrics (or test metrics if test_dataset provided)
        """
        # Try to resume from a previous checkpoint
        if auto_resume:
            self.resume_from_checkpoint()

        logger.info(
            f"Training epochs {self.start_epoch}→{self.config.train.num_epochs}"
        )
        logger.info(
            f"Train: {len(self.train_dataset)} snapshots, "
            f"Val: {len(self.val_dataset)} snapshots"
        )

        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self._evaluate(self.val_dataset)
            self.val_history.append(val_metrics)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Logging
            if epoch % self.config.train.log_interval == 0 or epoch == self.start_epoch:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{self.config.train.num_epochs} "
                    f"({epoch_time:.1f}s) | LR: {lr:.2e} | "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Acc: {train_metrics['accuracy']:.4f} "
                    f"DAMRR: {train_metrics['damrr']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Acc: {val_metrics['accuracy']:.4f} "
                    f"DAMRR: {val_metrics['damrr']:.4f}"
                )

            # TensorBoard logging
            if self.writer is not None:
                for key, val in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", val, epoch)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", val, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Early stopping check
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            # Periodic checkpoint (every N epochs)
            if epoch % self.checkpoint_interval == 0:
                self._save_periodic_checkpoint(epoch, val_metrics)

            if self.patience_counter >= self.config.train.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}"
                )
                break

        # Load best model
        self._load_best_checkpoint()

        # Plot training curves
        if self.train_history and self.val_history:
            exp = self.config.train.experiment_name or "training"
            plot_training_curves(
                self.train_history,
                self.val_history,
                save_path=os.path.join(
                    self.config.train.save_dir, f"training_curves_{exp}.png"
                ),
            )

        # Final test evaluation
        best_metrics = self.val_history[self.best_epoch - 1] if self.best_epoch > 0 else {}
        if self.test_dataset is not None:
            test_metrics = self._evaluate(self.test_dataset)
            logger.info("=" * 60)
            logger.info("TEST RESULTS:")
            logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {test_metrics['precision']:.4f}")
            logger.info(f"  DAMRR:     {test_metrics['damrr']:.4f}")
            logger.info(f"  Loss:      {test_metrics['loss']:.4f}")
            logger.info("=" * 60)
            return test_metrics

        return best_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        exp_name = self.config.train.experiment_name
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics": metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(
                self.config.train.save_dir, f"best_model_{exp_name}.pt"
            )
            torch.save(checkpoint, path)
            logger.info(f"Best model saved: {path} (epoch {epoch})")

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        exp_name = self.config.train.experiment_name
        path = os.path.join(
            self.config.train.save_dir, f"best_model_{exp_name}.pt"
        )
        # Fallback to old generic name
        if not os.path.exists(path):
            path = os.path.join(self.config.train.save_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best model from epoch {checkpoint['epoch']}"
            )

    @torch.no_grad()
    def get_predictions(
        self, dataset: StockTemporalDataset
    ) -> Tuple[list, list, list, list, list]:
        """
        Get model predictions for all snapshots in a dataset.

        Returns:
            Tuple of (dates, stock_codes, movement_preds, ranking_scores, actual_returns)
        """
        self.model.eval()
        self.model.reset_temporal_state()

        all_dates = []
        all_codes = []
        all_preds = []
        all_scores = []
        all_returns = []

        for idx in range(len(dataset)):
            snapshot = dataset[idx]
            snapshot = snapshot.to(self.device)

            outputs = self.model(
                node_features=snapshot.node_features,
                networks=snapshot.networks,
                stock_codes=snapshot.stock_codes,
            )

            preds = outputs["movement_logits"].argmax(dim=-1).cpu().numpy()
            scores = outputs["ranking_scores"].squeeze(-1).cpu().numpy()
            returns = snapshot.return_labels.cpu().numpy()

            all_dates.append(snapshot.date)
            all_codes.append(snapshot.stock_codes)
            all_preds.append(preds)
            all_scores.append(scores)
            all_returns.append(returns)

        return all_dates, all_codes, all_preds, all_scores, all_returns


def train_mst_gnn(
    config: Config,
    train_dataset: StockTemporalDataset,
    val_dataset: StockTemporalDataset,
    test_dataset: Optional[StockTemporalDataset] = None,
) -> Tuple[MSTGNN, Dict[str, float]]:
    """
    Convenience function to train MST-GNN.

    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset

    Returns:
        Tuple of (trained model, test metrics)
    """
    set_seed(config.train.seed)

    model = MSTGNN.from_config(config)
    trainer = Trainer(model, config, train_dataset, val_dataset, test_dataset)
    metrics = trainer.train()

    return model, metrics
