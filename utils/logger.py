"""
Logging utilities for ZeroDay-DRL framework.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "ZeroDay-DRL",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """
    Specialized logger for training metrics.
    """

    def __init__(self, log_dir: str = "logs/training"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.metrics = {
            'episode': [],
            'reward': [],
            'loss': [],
            'accuracy': [],
            'epsilon': [],
            'detection_rate': [],
            'false_positive_rate': []
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = os.path.join(log_dir, f'training_metrics_{timestamp}.csv')

    def log(self, **kwargs):
        """Log training metrics."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def save(self):
        """Save metrics to CSV file."""
        import csv

        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())

            max_len = max(len(v) for v in self.metrics.values())
            for i in range(max_len):
                row = []
                for key in self.metrics.keys():
                    if i < len(self.metrics[key]):
                        row.append(self.metrics[key][i])
                    else:
                        row.append('')
                writer.writerow(row)

    def get_recent(self, key: str, n: int = 10) -> list:
        """Get last n values for a metric."""
        if key in self.metrics:
            return self.metrics[key][-n:]
        return []
