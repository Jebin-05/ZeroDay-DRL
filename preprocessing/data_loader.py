"""
Data Loader for ZeroDay-DRL Framework.
Handles loading and preprocessing of IoT botnet datasets.
"""

import numpy as np
import os
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from .feature_extractor import FeatureExtractor
from .synthetic_generator import SyntheticDataGenerator


class IoTBotnetDataset(Dataset):
    """
    PyTorch Dataset for IoT Botnet traffic data.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Initialize dataset.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            transform: Optional transform to apply
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class FewShotEpisodeDataset(Dataset):
    """
    Dataset for few-shot episodic training.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100
    ):
        """
        Initialize few-shot dataset.

        Args:
            features: Feature array
            labels: Label array
            n_way: Number of classes per episode
            k_shot: Support samples per class
            n_query: Query samples per class
            n_episodes: Number of episodes to generate
        """
        self.features = features
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes

        # Organize samples by class
        self.class_indices = {}
        for label in np.unique(labels):
            self.class_indices[label] = np.where(labels == label)[0]

        # Pre-generate episodes
        self.episodes = self._generate_episodes()

    def _generate_episodes(self) -> List[Dict]:
        """Generate all episodes."""
        episodes = []

        for _ in range(self.n_episodes):
            # Select classes (always 0=normal, 1=botnet for binary)
            classes = list(self.class_indices.keys())[:self.n_way]

            support_features = []
            support_labels = []
            query_features = []
            query_labels = []

            for class_idx, cls in enumerate(classes):
                indices = self.class_indices[cls]
                selected = np.random.choice(
                    indices,
                    size=self.k_shot + self.n_query,
                    replace=len(indices) < self.k_shot + self.n_query
                )

                support_idx = selected[:self.k_shot]
                query_idx = selected[self.k_shot:]

                support_features.append(self.features[support_idx])
                support_labels.extend([class_idx] * self.k_shot)

                query_features.append(self.features[query_idx])
                query_labels.extend([class_idx] * self.n_query)

            episode = {
                'support_features': torch.FloatTensor(np.vstack(support_features)),
                'support_labels': torch.LongTensor(support_labels),
                'query_features': torch.FloatTensor(np.vstack(query_features)),
                'query_labels': torch.LongTensor(query_labels)
            }
            episodes.append(episode)

        return episodes

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> Dict:
        return self.episodes[idx]


class DataLoader:
    """
    Main data loader class for the ZeroDay-DRL framework.
    """

    def __init__(self, config: Dict):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_extractor = FeatureExtractor(
            window_size=config['features']['window_size'],
            slide_step=config['features']['slide_step'],
            feature_dim=config['features']['feature_dim']
        )
        self.generator = SyntheticDataGenerator(config['training']['seed'])

        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None
        self.test_features = None
        self.test_labels = None
        self.norm_params = None

    def load_synthetic_data(
        self,
        num_normal: int = 2000,
        num_botnet: int = 2000
    ) -> Dict[str, np.ndarray]:
        """
        Generate and load synthetic data.

        Args:
            num_normal: Number of normal flows
            num_botnet: Number of botnet flows

        Returns:
            Dictionary with train/val/test splits
        """
        # Generate synthetic data
        train_packets, test_packets = self.generator.generate_dataset(
            num_normal=num_normal,
            num_botnet=num_botnet,
            include_zero_day=True
        )

        # Extract features from training packets
        normal_packets = [p for p in train_packets if p['label'] == 0]
        botnet_packets = [p for p in train_packets if p['label'] == 1]

        normal_features, normal_labels = self.feature_extractor.extract_from_packets(
            normal_packets, label=0
        )
        botnet_features, botnet_labels = self.feature_extractor.extract_from_packets(
            botnet_packets, label=1
        )

        # Combine and shuffle
        all_features = np.vstack([normal_features, botnet_features])
        all_labels = np.concatenate([normal_labels, botnet_labels])

        indices = np.random.permutation(len(all_labels))
        all_features = all_features[indices]
        all_labels = all_labels[indices]

        # Split data
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']

        n_train = int(len(all_labels) * train_split)
        n_val = int(len(all_labels) * val_split)

        self.train_features = all_features[:n_train]
        self.train_labels = all_labels[:n_train]
        self.val_features = all_features[n_train:n_train + n_val]
        self.val_labels = all_labels[n_train:n_train + n_val]

        # Process test data (including zero-day)
        test_normal = [p for p in test_packets if p['label'] == 0]
        test_botnet = [p for p in test_packets if p['label'] == 1]

        test_normal_features, test_normal_labels = self.feature_extractor.extract_from_packets(
            test_normal, label=0
        )
        test_botnet_features, test_botnet_labels = self.feature_extractor.extract_from_packets(
            test_botnet, label=1
        )

        self.test_features = np.vstack([test_normal_features, test_botnet_features])
        self.test_labels = np.concatenate([test_normal_labels, test_botnet_labels])

        # Normalize features
        if self.config['features']['normalize']:
            self.train_features = self.feature_extractor.normalize(self.train_features, fit=True)
            self.val_features = self.feature_extractor.normalize(self.val_features, fit=False)
            self.test_features = self.feature_extractor.normalize(self.test_features, fit=False)

        return {
            'train': (self.train_features, self.train_labels),
            'val': (self.val_features, self.val_labels),
            'test': (self.test_features, self.test_labels)
        }

    def load_csv_data(self, csv_path: str) -> Dict[str, np.ndarray]:
        """
        Load data from CSV file (for real datasets like IoT-23).

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with train/val/test splits
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Expected columns (adjust based on actual dataset)
        feature_cols = [col for col in df.columns if col not in ['label', 'Label', 'attack']]
        label_col = 'label' if 'label' in df.columns else 'Label'

        features = df[feature_cols].values
        labels = df[label_col].values

        # Convert labels to binary (0=normal, 1=attack)
        if labels.dtype == object:
            labels = (labels != 'normal').astype(int)

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        # Split data
        indices = np.random.permutation(len(labels))
        features = features[indices]
        labels = labels[indices]

        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']

        n_train = int(len(labels) * train_split)
        n_val = int(len(labels) * val_split)

        self.train_features = features[:n_train]
        self.train_labels = labels[:n_train]
        self.val_features = features[n_train:n_train + n_val]
        self.val_labels = labels[n_train:n_train + n_val]
        self.test_features = features[n_train + n_val:]
        self.test_labels = labels[n_train + n_val:]

        # Normalize
        if self.config['features']['normalize']:
            from utils.helpers import normalize_features, apply_normalization
            self.train_features, self.norm_params = normalize_features(
                self.train_features, method='minmax'
            )
            self.val_features = apply_normalization(
                self.val_features, self.norm_params, method='minmax'
            )
            self.test_features = apply_normalization(
                self.test_features, self.norm_params, method='minmax'
            )

        return {
            'train': (self.train_features, self.train_labels),
            'val': (self.val_features, self.val_labels),
            'test': (self.test_features, self.test_labels)
        }

    def get_dataloaders(
        self,
        batch_size: int = 64
    ) -> Dict[str, TorchDataLoader]:
        """
        Get PyTorch DataLoaders for training.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of DataLoaders
        """
        if self.train_features is None:
            raise ValueError("Data not loaded. Call load_synthetic_data() first.")

        train_dataset = IoTBotnetDataset(self.train_features, self.train_labels)
        val_dataset = IoTBotnetDataset(self.val_features, self.val_labels)
        test_dataset = IoTBotnetDataset(self.test_features, self.test_labels)

        return {
            'train': TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

    def get_few_shot_loader(
        self,
        n_episodes: int = 100,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15
    ) -> TorchDataLoader:
        """
        Get DataLoader for few-shot training.

        Args:
            n_episodes: Number of episodes
            n_way: Number of classes
            k_shot: Support samples
            n_query: Query samples

        Returns:
            Few-shot DataLoader
        """
        if self.train_features is None:
            raise ValueError("Data not loaded. Call load_synthetic_data() first.")

        dataset = FewShotEpisodeDataset(
            self.train_features,
            self.train_labels,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes
        )

        return TorchDataLoader(dataset, batch_size=1, shuffle=True)

    def get_drl_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data formatted for DRL environment.

        Returns:
            Tuple of (features, labels)
        """
        if self.train_features is None:
            raise ValueError("Data not loaded. Call load_synthetic_data() first.")

        return self.train_features, self.train_labels

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.train_features is None:
            return {}

        return {
            'train_samples': len(self.train_labels),
            'val_samples': len(self.val_labels),
            'test_samples': len(self.test_labels),
            'train_normal': int((self.train_labels == 0).sum()),
            'train_botnet': int((self.train_labels == 1).sum()),
            'feature_dim': self.train_features.shape[1],
            'feature_means': self.train_features.mean(axis=0).tolist(),
            'feature_stds': self.train_features.std(axis=0).tolist()
        }
