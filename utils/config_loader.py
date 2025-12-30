"""
Configuration loader for ZeroDay-DRL framework.
Handles YAML configuration files and provides easy access to parameters.
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing all configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """
    Get the computation device based on configuration and availability.

    Args:
        config: Configuration dictionary (optional)

    Returns:
        torch.device object
    """
    if config is None:
        device_str = "cpu"
    else:
        device_str = config.get('training', {}).get('device', 'cpu')

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update configuration with new values.

    Args:
        config: Original configuration dictionary
        updates: Dictionary with updates

    Returns:
        Updated configuration dictionary
    """
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config


class ConfigManager:
    """
    Singleton configuration manager for easy access throughout the application.
    """
    _instance = None
    _config = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def load(self, config_path: str = "configs/config.yaml"):
        """Load configuration from file."""
        self._config = load_config(config_path)
        return self._config

    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        return self._config

    def get(self, *keys, default=None):
        """
        Get nested configuration value.

        Args:
            *keys: Sequence of keys to traverse
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
