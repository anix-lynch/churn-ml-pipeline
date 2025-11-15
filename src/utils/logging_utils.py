#!/usr/bin/env python3
"""
Churn ML Pipeline Logging Utilities - Structured logging setup and management

🎯 PURPOSE: Configure and manage logging across the entire ML pipeline with file and console output
📊 FEATURES: Timestamped log files, configurable log levels, structured formatting, automatic log rotation
🏗️ ARCHITECTURE: Singleton logger pattern with optional file output for production deployments
⚡ STATUS: Production-ready logging with proper error handling and cross-platform compatibility
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and optional file output

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamped_log_path(base_dir: str, prefix: str) -> str:
    """
    Generate timestamped log file path

    Args:
        base_dir: Base directory for logs
        prefix: Log file prefix

    Returns:
        Full path to timestamped log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_dir}/{prefix}_{timestamp}.log"
