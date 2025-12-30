"""
Neural Network architectures for DRL agents.
Lightweight designs suitable for IoT deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class QNetwork(nn.Module):
    """
    Q-Network for DQN agent.
    Maps state to Q-values for each action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Light regularization
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch, state_dim)

        Returns:
            Q-values tensor of shape (batch, action_dim)
        """
        return self.network(state)

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, float]:
        """
        Get action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Tuple of (action, Q-value)
        """
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            with torch.no_grad():
                q_values = self.forward(state)
                q_value = q_values[0, action].item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0, action].item()

        return action, q_value


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    Separates state value and advantage estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        """
        Initialize Dueling Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        feature_layers = []
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        features = self.feature_extractor(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, float]:
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            with torch.no_grad():
                q_values = self.forward(state)
                q_value = q_values[0, action].item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0, action].item()

        return action, q_value


class PolicyNetwork(nn.Module):
    """
    Policy Network for PPO agent.
    Outputs action probabilities.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize Policy Network.

        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            Action distribution
        """
        features = self.features(state)
        action_logits = self.action_head(features)
        return torch.distributions.Categorical(logits=action_logits)

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob)
        """
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probs, entropy)
        """
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value Network for PPO agent.
    Estimates state value.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize Value Network.

        Args:
            state_dim: State dimension
            hidden_dims: Hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            State value
        """
        return self.network(state).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network.
    Shares feature extraction between policy and value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize Actor-Critic Network.

        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extractor
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Linear(prev_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)

        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0)

    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            Tuple of (action distribution, state value)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        dist = torch.distributions.Categorical(logits=action_logits)
        return dist, value

    def get_action_and_value(
        self,
        state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, and value.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        dist, value = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
