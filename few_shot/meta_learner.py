"""
Meta-Learner for Few-Shot IoT Botnet Detection.
Coordinates few-shot learning with online adaptation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque

from .prototypical_network import PrototypicalNetwork


class MetaLearner:
    """
    Meta-learner that manages few-shot classification
    and online adaptation to new botnet variants.
    """

    def __init__(
        self,
        input_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize meta-learner.

        Args:
            input_dim: Input feature dimension
            config: Configuration dictionary
            device: Computation device
        """
        self.config = config
        self.device = device
        self.input_dim = input_dim

        # Extract few-shot config
        fs_config = config.get('few_shot', {})

        self.embedding_dim = fs_config.get('embedding_dim', 64)
        self.hidden_dims = fs_config.get('hidden_dims', [128, 64])
        self.n_way = fs_config.get('n_way', 2)
        self.k_shot = fs_config.get('k_shot', 5)
        self.n_query = fs_config.get('n_query', 15)
        self.distance = fs_config.get('distance', 'euclidean')
        self.learning_rate = fs_config.get('learning_rate', 0.001)

        # Initialize prototypical network
        self.proto_net = PrototypicalNetwork(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            distance=self.distance
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.proto_net.parameters(),
            lr=self.learning_rate
        )

        # Experience buffer for online adaptation
        self.experience_buffer = {
            0: deque(maxlen=100),  # Normal samples
            1: deque(maxlen=100)   # Botnet samples
        }

        # Prototype memory
        self.prototype_memory = {}

        # Metrics
        self.adaptation_count = 0
        self.classification_history = []

    def episodic_train(
        self,
        dataloader,
        num_episodes: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train using episodic training.

        Args:
            dataloader: Few-shot episode dataloader
            num_episodes: Number of training episodes

        Returns:
            Training metrics
        """
        self.proto_net.train()

        losses = []
        accuracies = []

        for episode_idx, episode in enumerate(dataloader):
            if episode_idx >= num_episodes:
                break

            support_features = episode['support_features'].squeeze(0).to(self.device)
            support_labels = episode['support_labels'].squeeze(0).to(self.device)
            query_features = episode['query_features'].squeeze(0).to(self.device)
            query_labels = episode['query_labels'].squeeze(0).to(self.device)

            # Forward pass
            log_probs, prototypes = self.proto_net(
                support_features,
                support_labels,
                query_features
            )

            # Compute loss
            loss = F.nll_loss(log_probs, query_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proto_net.parameters(), 1.0)
            self.optimizer.step()

            # Compute accuracy
            preds = log_probs.argmax(dim=1)
            accuracy = (preds == query_labels).float().mean().item()

            losses.append(loss.item())
            accuracies.append(accuracy)

            if (episode_idx + 1) % 50 == 0:
                print(f"Episode {episode_idx + 1}/{num_episodes} - "
                      f"Loss: {np.mean(losses[-50:]):.4f}, "
                      f"Accuracy: {np.mean(accuracies[-50:]):.4f}")

        return {
            'losses': losses,
            'accuracies': accuracies
        }

    def initialize_prototypes(
        self,
        normal_samples: np.ndarray,
        botnet_samples: np.ndarray
    ):
        """
        Initialize prototypes from labeled samples.

        Args:
            normal_samples: Normal traffic samples
            botnet_samples: Botnet traffic samples
        """
        self.proto_net.eval()

        # Select k_shot samples from each class
        n_normal = min(self.k_shot, len(normal_samples))
        n_botnet = min(self.k_shot, len(botnet_samples))

        normal_idx = np.random.choice(len(normal_samples), n_normal, replace=False)
        botnet_idx = np.random.choice(len(botnet_samples), n_botnet, replace=False)

        support_features = np.vstack([
            normal_samples[normal_idx],
            botnet_samples[botnet_idx]
        ])
        support_labels = np.array([0] * n_normal + [1] * n_botnet)

        support_tensor = torch.FloatTensor(support_features).to(self.device)
        labels_tensor = torch.LongTensor(support_labels).to(self.device)

        self.proto_net.update_prototypes(support_tensor, labels_tensor)

    def classify(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify samples using current prototypes.

        Args:
            features: Input features

        Returns:
            Tuple of (predictions, confidences)
        """
        self.proto_net.eval()

        features_tensor = torch.FloatTensor(features).to(self.device)

        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)

        predictions, confidences = self.proto_net.predict(features_tensor)

        return predictions.cpu().numpy(), confidences.cpu().numpy()

    def classify_single(
        self,
        features: np.ndarray
    ) -> Tuple[int, float]:
        """
        Classify a single sample.

        Args:
            features: Single sample features

        Returns:
            Tuple of (prediction, confidence)
        """
        preds, confs = self.classify(features.reshape(1, -1))
        return int(preds[0]), float(confs[0])

    def add_to_experience(
        self,
        features: np.ndarray,
        label: int
    ):
        """
        Add sample to experience buffer.

        Args:
            features: Sample features
            label: Sample label
        """
        self.experience_buffer[label].append(features)

    def adapt(self, adaptation_rate: float = 0.1):
        """
        Adapt prototypes using experience buffer.

        Args:
            adaptation_rate: Rate of prototype update
        """
        if len(self.experience_buffer[0]) < 3 and len(self.experience_buffer[1]) < 3:
            return

        self.proto_net.eval()

        # Gather recent experiences
        features_list = []
        labels_list = []

        for label, buffer in self.experience_buffer.items():
            if len(buffer) > 0:
                class_features = np.array(list(buffer))
                features_list.append(class_features)
                labels_list.extend([label] * len(class_features))

        if len(features_list) == 0:
            return

        features = np.vstack(features_list)
        labels = np.array(labels_list)

        features_tensor = torch.FloatTensor(features).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)

        # Adapt prototypes
        self.proto_net.adapt_to_new_samples(
            features_tensor,
            labels_tensor,
            adaptation_rate
        )

        self.adaptation_count += 1

    def get_prototype_distances(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Get distances from sample to each prototype.

        Args:
            features: Sample features

        Returns:
            Dictionary of distances
        """
        self.proto_net.eval()

        features_tensor = torch.FloatTensor(features).to(self.device)
        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.proto_net.embedding_net(features_tensor)
            distances = self.proto_net.compute_distances(
                embeddings,
                self.proto_net.prototypes
            )

        return {
            'normal_distance': distances[0, 0].item(),
            'botnet_distance': distances[0, 1].item() if distances.shape[1] > 1 else 0
        }

    def get_embedding(self, features: np.ndarray) -> np.ndarray:
        """
        Get embedding representation of features.

        Args:
            features: Input features

        Returns:
            Embedding vector
        """
        features_tensor = torch.FloatTensor(features).to(self.device)
        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)

        return self.proto_net.get_embedding(features_tensor).cpu().numpy()

    def compute_novelty_score(self, features: np.ndarray) -> float:
        """
        Compute novelty score for potential zero-day detection.
        High score indicates sample is far from known prototypes.

        Args:
            features: Sample features

        Returns:
            Novelty score (0-1)
        """
        distances = self.get_prototype_distances(features)
        min_distance = min(distances.values())

        # Normalize using learned threshold
        novelty = 1 - np.exp(-min_distance / 2)
        return float(novelty)

    def save(self, path: str):
        """Save meta-learner state."""
        torch.save({
            'proto_net': self.proto_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'prototypes': self.proto_net.prototypes,
            'prototype_labels': self.proto_net.prototype_labels,
            'adaptation_count': self.adaptation_count,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load meta-learner state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.proto_net.load_state_dict(checkpoint['proto_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.proto_net.prototypes = checkpoint['prototypes']
        self.proto_net.prototype_labels = checkpoint['prototype_labels']
        self.adaptation_count = checkpoint['adaptation_count']

    def get_metrics(self) -> Dict:
        """Get meta-learner metrics."""
        return {
            'adaptation_count': self.adaptation_count,
            'buffer_normal': len(self.experience_buffer[0]),
            'buffer_botnet': len(self.experience_buffer[1]),
            'has_prototypes': self.proto_net.prototypes is not None
        }
