"""
IoT-23 Dataset Loader for ZeroDay-DRL Framework.
Handles the real IoT-23 botnet dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class IoT23Dataset(Dataset):
    """PyTorch Dataset for IoT-23 data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class IoT23DataLoader:
    """
    Data loader for IoT-23 dataset.
    Handles both cleaned_data.csv and iot23_combined.csv.
    """

    # Early-stage attack labels (for focused detection)
    EARLY_STAGE_LABELS = [
        'PartOfAHorizontalPortScan',
        'C&C',
        'C&C-HeartBeat',
        'C&C-FileDownload',
        'C&C-Torii',
        'C&C-Mirai',
        'C&C-HeartBeat-FileDownload'
    ]

    # All attack labels
    ATTACK_LABELS = EARLY_STAGE_LABELS + [
        'Attack',
        'DDoS',
        'Okiru',
        'FileDownload'
    ]

    # Normal labels
    NORMAL_LABELS = ['Benign', '-   Benign   -']

    def __init__(self, config: Dict, data_dir: str = 'data'):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary
            data_dir: Directory containing data files
        """
        self.config = config
        self.data_dir = data_dir

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Data storage
        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None
        self.test_features = None
        self.test_labels = None

        # Feature columns (excluding label and non-numeric)
        self.feature_columns = None
        self.label_column = 'label'

    def load_cleaned_data(
        self,
        filename: str = 'cleaned_data.csv',
        early_stage_only: bool = False,
        sample_size: Optional[int] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load the cleaned IoT-23 dataset.

        Args:
            filename: CSV filename
            early_stage_only: If True, only include early-stage attacks
            sample_size: Optional sample size limit

        Returns:
            Dictionary with train/val/test splits
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Convert labels to binary (Normal=0, Attack=1)
        df['binary_label'] = df[self.label_column].apply(self._label_to_binary)

        # Filter to early-stage if requested
        if early_stage_only:
            mask = df[self.label_column].isin(self.NORMAL_LABELS + self.EARLY_STAGE_LABELS)
            df = df[mask]
            print(f"Filtered to early-stage: {len(df)} samples")

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled to: {len(df)} samples")

        # Get feature columns (numeric only, excluding labels)
        self.feature_columns = [
            col for col in df.columns
            if col not in ['label', 'binary_label', 'Unnamed: 0', 'ts', 'id.orig_h']
            and df[col].dtype in ['int64', 'float64']
        ]

        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns[:5]}...")

        # Extract features and labels
        features = df[self.feature_columns].values.astype(np.float32)
        labels = df['binary_label'].values.astype(np.int64)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: Normal={counts[0]}, Botnet={counts[1] if len(counts) > 1 else 0}")

        # Split data
        return self._split_data(features, labels)

    def load_full_dataset(
        self,
        filename: str = 'iot23_combined.csv',
        sample_size: int = 100000
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load the full IoT-23 dataset (with sampling for memory efficiency).

        Args:
            filename: CSV filename
            sample_size: Maximum samples to load

        Returns:
            Dictionary with train/val/test splits
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        print(f"Loading data from {filepath} (sampling {sample_size} rows)...")

        # Count total rows
        total_rows = sum(1 for _ in open(filepath)) - 1
        print(f"Total rows in file: {total_rows}")

        # Calculate skip probability for sampling
        if total_rows > sample_size:
            skip_prob = 1 - (sample_size / total_rows)
            skiprows = lambda i: i > 0 and np.random.random() < skip_prob
            df = pd.read_csv(filepath, skiprows=skiprows)
        else:
            df = pd.read_csv(filepath)

        print(f"Loaded samples: {len(df)}")

        # Process same as cleaned data
        df['binary_label'] = df[self.label_column].apply(self._label_to_binary)

        # Get feature columns
        self.feature_columns = [
            col for col in df.columns
            if col not in ['label', 'binary_label', 'Unnamed: 0', 'ts', 'id.orig_h']
            and df[col].dtype in ['int64', 'float64']
        ]

        features = df[self.feature_columns].values.astype(np.float32)
        labels = df['binary_label'].values.astype(np.int64)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

        return self._split_data(features, labels)

    def _label_to_binary(self, label: str) -> int:
        """Convert string label to binary (0=normal, 1=attack)."""
        if label in self.NORMAL_LABELS:
            return 0
        else:
            return 1

    def _split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets."""
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']

        # Shuffle
        indices = np.random.permutation(len(labels))
        features = features[indices]
        labels = labels[indices]

        n_train = int(len(labels) * train_split)
        n_val = int(len(labels) * val_split)

        self.train_features = features[:n_train]
        self.train_labels = labels[:n_train]
        self.val_features = features[n_train:n_train + n_val]
        self.val_labels = labels[n_train:n_train + n_val]
        self.test_features = features[n_train + n_val:]
        self.test_labels = labels[n_train + n_val:]

        # Normalize features
        if self.config['features'].get('normalize', True):
            self.train_features = self.scaler.fit_transform(self.train_features)
            self.val_features = self.scaler.transform(self.val_features)
            self.test_features = self.scaler.transform(self.test_features)

        print(f"Train: {len(self.train_labels)} samples")
        print(f"Val: {len(self.val_labels)} samples")
        print(f"Test: {len(self.test_labels)} samples")

        return {
            'train': (self.train_features, self.train_labels),
            'val': (self.val_features, self.val_labels),
            'test': (self.test_features, self.test_labels)
        }

    def get_dataloaders(self, batch_size: int = 64) -> Dict[str, TorchDataLoader]:
        """Get PyTorch DataLoaders."""
        if self.train_features is None:
            raise ValueError("Data not loaded. Call load_cleaned_data() first.")

        train_dataset = IoT23Dataset(self.train_features, self.train_labels)
        val_dataset = IoT23Dataset(self.val_features, self.val_labels)
        test_dataset = IoT23Dataset(self.test_features, self.test_labels)

        return {
            'train': TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

    def get_few_shot_samples(
        self,
        n_normal: int = 10,
        n_attack: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get samples for few-shot prototype initialization.

        Args:
            n_normal: Number of normal samples
            n_attack: Number of attack samples

        Returns:
            Tuple of (normal_samples, attack_samples)
        """
        if self.train_features is None:
            raise ValueError("Data not loaded.")

        normal_mask = self.train_labels == 0
        attack_mask = self.train_labels == 1

        normal_samples = self.train_features[normal_mask][:n_normal]
        attack_samples = self.train_features[attack_mask][:n_attack]

        return normal_samples, attack_samples

    def get_attack_types(self, filename: str = 'cleaned_data.csv') -> Dict[str, np.ndarray]:
        """
        Get samples grouped by attack type for zero-day testing.

        Args:
            filename: CSV filename

        Returns:
            Dictionary mapping attack type to feature arrays
        """
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)

        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.columns
                if col not in ['label', 'Unnamed: 0', 'ts', 'id.orig_h']
                and df[col].dtype in ['int64', 'float64']
            ]

        attack_data = {}

        for label in df['label'].unique():
            if label not in self.NORMAL_LABELS:
                mask = df['label'] == label
                features = df.loc[mask, self.feature_columns].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0)

                if hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
                    features = self.scaler.transform(features)

                attack_data[label] = features

        return attack_data

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.train_features is None:
            return {}

        return {
            'train_samples': len(self.train_labels),
            'val_samples': len(self.val_labels),
            'test_samples': len(self.test_labels),
            'train_normal': int((self.train_labels == 0).sum()),
            'train_attack': int((self.train_labels == 1).sum()),
            'feature_dim': self.train_features.shape[1],
            'feature_names': self.feature_columns
        }
