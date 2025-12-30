"""
Helper utilities for ZeroDay-DRL framework.
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    additional_info: Dict[str, Any] = None
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch/episode
        path: Save path
        additional_info: Additional information to save
    """
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, path)


def load_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model to load weights into
        path: Checkpoint path
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model to

    Returns:
        Dictionary with checkpoint info
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def moving_average(values: list, window: int = 10) -> list:
    """
    Calculate moving average.

    Args:
        values: List of values
        window: Window size

    Returns:
        List of moving average values
    """
    if len(values) < window:
        return values

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def normalize_features(features: np.ndarray, method: str = 'minmax') -> tuple:
    """
    Normalize features.

    Args:
        features: Feature array
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        Tuple of (normalized features, normalization parameters)
    """
    if method == 'minmax':
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        normalized = (features - min_vals) / range_vals
        params = {'min': min_vals, 'max': max_vals}

    elif method == 'zscore':
        mean_vals = features.mean(axis=0)
        std_vals = features.std(axis=0)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        normalized = (features - mean_vals) / std_vals
        params = {'mean': mean_vals, 'std': std_vals}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def apply_normalization(
    features: np.ndarray,
    params: Dict[str, np.ndarray],
    method: str = 'minmax'
) -> np.ndarray:
    """
    Apply pre-computed normalization to features.

    Args:
        features: Feature array
        params: Normalization parameters
        method: Normalization method

    Returns:
        Normalized features
    """
    if method == 'minmax':
        range_vals = params['max'] - params['min']
        range_vals[range_vals == 0] = 1
        return (features - params['min']) / range_vals

    elif method == 'zscore':
        std_vals = params['std'].copy()
        std_vals[std_vals == 0] = 1
        return (features - params['mean']) / std_vals

    else:
        raise ValueError(f"Unknown normalization method: {method}")


class EarlyStopping:
    """
    Early stopping utility for training.
    """

    def __init__(self, patience: int = 50, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
