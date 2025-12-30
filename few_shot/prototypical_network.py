"""
Prototypical Networks for Few-Shot IoT Botnet Detection.
Enables rapid adaptation to unseen (zero-day) botnet variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional


class EmbeddingNetwork(nn.Module):
    """
    Embedding network for prototypical networks.
    Maps input features to embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize embedding network.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super(EmbeddingNetwork, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Output embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        return self.network(x)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification.
    Learns to classify by comparing to class prototypes.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        distance: str = 'euclidean'
    ):
        """
        Initialize Prototypical Network.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            distance: Distance metric ('euclidean' or 'cosine')
        """
        super(PrototypicalNetwork, self).__init__()

        self.embedding_net = EmbeddingNetwork(
            input_dim, embedding_dim, hidden_dims
        )
        self.distance = distance
        self.embedding_dim = embedding_dim

        # Store prototypes for inference
        self.prototypes = None
        self.prototype_labels = None

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set.

        Args:
            support_features: Support set features (n_support, input_dim)
            support_labels: Support set labels (n_support,)

        Returns:
            Prototypes tensor (n_classes, embedding_dim)
        """
        # Get embeddings
        embeddings = self.embedding_net(support_features)

        # Get unique classes
        classes = torch.unique(support_labels)
        n_classes = len(classes)

        # Compute prototype for each class
        prototypes = torch.zeros(n_classes, self.embedding_dim).to(embeddings.device)

        for i, c in enumerate(classes):
            mask = support_labels == c
            class_embeddings = embeddings[mask]
            prototypes[i] = class_embeddings.mean(dim=0)

        return prototypes

    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances from queries to prototypes.

        Args:
            query_embeddings: Query embeddings (n_query, embedding_dim)
            prototypes: Class prototypes (n_classes, embedding_dim)

        Returns:
            Distance tensor (n_query, n_classes)
        """
        if self.distance == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance == 'cosine':
            # Cosine distance (1 - cosine similarity)
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        return distances

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for episodic training.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features

        Returns:
            Tuple of (log_probabilities, prototypes)
        """
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_features, support_labels)

        # Get query embeddings
        query_embeddings = self.embedding_net(query_features)

        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)

        # Convert distances to log probabilities (negative distance as logits)
        log_probs = F.log_softmax(-distances, dim=1)

        return log_probs, prototypes

    def update_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ):
        """
        Update stored prototypes for inference.

        Args:
            support_features: Support set features
            support_labels: Support set labels
        """
        with torch.no_grad():
            self.prototypes = self.compute_prototypes(
                support_features, support_labels
            )
            self.prototype_labels = torch.unique(support_labels)

    def predict(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using stored prototypes.

        Args:
            features: Input features

        Returns:
            Tuple of (predictions, confidences)
        """
        if self.prototypes is None:
            raise ValueError("No prototypes stored. Call update_prototypes first.")

        with torch.no_grad():
            embeddings = self.embedding_net(features)
            distances = self.compute_distances(embeddings, self.prototypes)
            probs = F.softmax(-distances, dim=1)

            predictions = probs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]

        return predictions, confidences

    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Get embedding for features."""
        with torch.no_grad():
            return self.embedding_net(features)

    def adapt_to_new_samples(
        self,
        new_features: torch.Tensor,
        new_labels: torch.Tensor,
        adaptation_rate: float = 0.1
    ):
        """
        Adapt prototypes with new samples (online adaptation).

        Args:
            new_features: New sample features
            new_labels: New sample labels
            adaptation_rate: Rate of prototype update
        """
        if self.prototypes is None:
            self.update_prototypes(new_features, new_labels)
            return

        with torch.no_grad():
            new_embeddings = self.embedding_net(new_features)

            for label in torch.unique(new_labels):
                label_idx = (self.prototype_labels == label).nonzero(as_tuple=True)[0]
                if len(label_idx) > 0:
                    mask = new_labels == label
                    new_proto = new_embeddings[mask].mean(dim=0)

                    # Exponential moving average update
                    self.prototypes[label_idx] = (
                        (1 - adaptation_rate) * self.prototypes[label_idx] +
                        adaptation_rate * new_proto
                    )


class PrototypicalTrainer:
    """
    Trainer for Prototypical Networks using episodic training.
    """

    def __init__(
        self,
        model: PrototypicalNetwork,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize trainer.

        Args:
            model: Prototypical network
            config: Configuration dictionary
            device: Computation device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        few_shot_config = config.get('few_shot', {})
        self.learning_rate = few_shot_config.get('learning_rate', 0.001)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )

        # Metrics
        self.losses = []
        self.accuracies = []

    def train_episode(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Train on a single episode.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features
            query_labels: Query set labels

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()

        # Forward pass
        log_probs, _ = self.model(
            support_features.to(self.device),
            support_labels.to(self.device),
            query_features.to(self.device)
        )

        # Compute loss (negative log likelihood)
        loss = F.nll_loss(log_probs, query_labels.to(self.device))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Compute accuracy
        predictions = log_probs.argmax(dim=1)
        accuracy = (predictions == query_labels.to(self.device)).float().mean().item()

        # Track metrics
        self.losses.append(loss.item())
        self.accuracies.append(accuracy)

        return loss.item(), accuracy

    def evaluate_episode(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Evaluate on a single episode.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features
            query_labels: Query set labels

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()

        with torch.no_grad():
            log_probs, _ = self.model(
                support_features.to(self.device),
                support_labels.to(self.device),
                query_features.to(self.device)
            )

            loss = F.nll_loss(log_probs, query_labels.to(self.device))
            predictions = log_probs.argmax(dim=1)
            accuracy = (predictions == query_labels.to(self.device)).float().mean().item()

        return loss.item(), accuracy

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_accuracy': np.mean(self.accuracies[-100:]) if self.accuracies else 0,
            'total_episodes': len(self.losses)
        }

    def save(self, path: str):
        """Save model."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
