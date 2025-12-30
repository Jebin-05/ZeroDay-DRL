"""
PPO Agent implementation for IoT Intrusion Detection.
Proximal Policy Optimization with clipped objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List

from .networks import ActorCriticNetwork


class RolloutBuffer:
    """
    Buffer for storing rollout experiences for PPO.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization agent for intrusion detection.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO agent.

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

        # Extract PPO config
        drl_config = config.get('drl', {})
        ppo_config = drl_config.get('ppo', {})

        # Hyperparameters
        self.gamma = drl_config.get('gamma', 0.99)
        self.learning_rate = drl_config.get('learning_rate', 0.0003)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.value_coef = ppo_config.get('value_coef', 0.5)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.num_epochs = ppo_config.get('num_epochs', 4)
        self.num_steps = ppo_config.get('num_steps', 128)
        self.batch_size = drl_config.get('batch_size', 64)

        # Network
        hidden_dims = drl_config.get('hidden_dims', [128, 64])
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training state
        self.train_step = 0

        # Metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """
        Select action from policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Tuple of (action, confidence, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, value = self.network(state_tensor)

            if training:
                action = dist.sample()
            else:
                action = dist.probs.argmax()

            log_prob = dist.log_prob(action)
            confidence = dist.probs[0, action].item()

        return action.item(), confidence, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """
        Store transition in rollout buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Action log probability
            value: State value
            done: Episode done flag
        """
        self.buffer.push(state, action, reward, log_prob, value, done)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of values
            dones: List of done flags
            next_value: Value of next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0

        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        values.append(next_value.item() if isinstance(next_value, torch.Tensor) else next_value)

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        Update policy using collected rollouts.

        Args:
            next_state: State after rollout

        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}

        # Get next value for GAE
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value
        )

        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)

        # Training epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.num_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current policy outputs
                dist, values = self.network(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Clear buffer
        self.buffer.clear()

        # Track metrics
        n_updates = self.num_epochs * (len(states) // self.batch_size + 1)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)

        self.train_step += 1

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }

    def get_confidence(self, state: np.ndarray) -> float:
        """
        Get detection confidence for current state.

        Args:
            state: State array

        Returns:
            Confidence value (0-1)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _ = self.network(state_tensor)
            confidence = dist.probs.max().item()

        return confidence

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a state.

        Args:
            state: State array

        Returns:
            Action probabilities
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _ = self.network(state_tensor)
            probs = dist.probs.cpu().numpy()[0]

        return probs

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'avg_entropy': np.mean(self.entropies[-100:]) if self.entropies else 0,
            'train_steps': self.train_step
        }
