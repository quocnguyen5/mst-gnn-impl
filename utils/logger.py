"""
Logging Setup
=============
Configures logging and TensorBoard for experiment tracking.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(
    name: str = "mst_gnn",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and optional console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log")
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def setup_tensorboard(log_dir: str = "runs"):
    """
    Set up TensorBoard writer.

    Args:
        log_dir: TensorBoard log directory

    Returns:
        SummaryWriter or None if tensorboard is not available
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(log_dir, timestamp))
        return writer
    except ImportError:
        logging.warning("TensorBoard not available. Install with: pip install tensorboard")
        return None
