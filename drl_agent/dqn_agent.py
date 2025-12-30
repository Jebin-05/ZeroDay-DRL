"""
DQN Agent implementation for IoT Intrusion Detection.
Implements Double DQN with optional Dueling architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy

from .networks import QNetwork, DuelingQNetwork
from env.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent for intrusion detection.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            config: Configuration dictionary
            device: Computation device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        # Extract DRL config
        drl_config = config.get('drl', {})

        # Hyperparameters
        self.gamma = drl_config.get('gamma', 0.99)
        self.learning_rate = drl_config.get('learning_rate', 0.001)
        self.batch_size = drl_config.get('batch_size', 64)
        self.buffer_size = drl_config.get('buffer_size', 10000)
        self.target_update_freq = drl_config.get('target_update_freq', 100)
        self.epsilon = drl_config.get('epsilon_start', 1.0)
        self.epsilon_end = drl_config.get('epsilon_end', 0.01)
        self.epsilon_decay = drl_config.get('epsilon_decay', 0.995)

        # Network architecture
        hidden_dims = drl_config.get('hidden_dims', [128, 64, 32])

        # Initialize networks (use dueling architecture for better performance)
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.target_network = DuelingQNetwork(
            state_dim, action_dim, hidden_dims
        ).to(device)

        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Training state
        self.train_step = 0
        self.update_count = 0

        # Metrics
        self.losses = []
        self.q_values = []

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Tuple of (action, confidence)
        """
        epsilon = self.epsilon if training else 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

            if np.random.random() < epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = q_values.argmax(dim=1).item()

            # Calculate confidence (softmax of Q-values)
            probs = torch.softmax(q_values, dim=1)
            confidence = probs[0, action].item()

        return action, confidence

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step_fn(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Track metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q.mean().item())

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()

        return loss.item()

    def _update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.update_count += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state.

        Args:
            state: State array

        Returns:
            Q-values array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.cpu().numpy()[0]

    def get_confidence(self, state: np.ndarray) -> float:
        """
        Get detection confidence for current state.

        Args:
            state: State array

        Returns:
            Confidence value (0-1)
        """
        q_values = self.get_q_values(state)
        probs = np.exp(q_values) / np.sum(np.exp(q_values))
        return float(np.max(probs))

    def save(self, path: str):
        """
        Save agent state.

        Args:
            path: Save path
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'config': self.config
        }, path)

    def load(self, path: str):
        """
        Load agent state.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'epsilon': self.epsilon,
            'train_steps': self.train_step,
            'target_updates': self.update_count,
            'buffer_size': len(self.replay_buffer)
        }


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent with additional improvements.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu'),
        use_prioritized_replay: bool = True
    ):
        """
        Initialize Double DQN agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Configuration
            device: Device
            use_prioritized_replay: Whether to use PER
        """
        super().__init__(state_dim, action_dim, config, device)

        # Replace replay buffer with prioritized version if requested
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                alpha=0.6,
                beta_start=0.4
            )
            self.use_per = True
        else:
            self.use_per = False

    def train_step_fn(self) -> Optional[float]:
        """Training step with PER support."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        if self.use_per:
            # Sample with priorities
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(len(states)).to(self.device)
            indices = None

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Weighted loss for PER
        td_errors = torch.abs(current_q - target_q).detach()
        loss = (weights * (current_q - target_q) ** 2).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        if self.use_per and indices is not None:
            self.replay_buffer.update_priorities(
                indices,
                td_errors.cpu().numpy()
            )

        # Track metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q.mean().item())

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()

        return loss.item()
